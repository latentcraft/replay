from __future__ import annotations

import os
import os.path as osp
import torch
import torch.nn.functional as F
import logging
import warnings
from PIL import Image

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from .qwen2_vl.model import ensure_image_url
from ..smp import get_gpu_memory, listinstr


class LocalSmoothQwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    
    @staticmethod
    def _get_gpu_idx():
        """Get current GPU index from environment or device"""
        # Try LOCAL_RANK first (set by torchrun)
        if 'LOCAL_RANK' in os.environ:
            return int(os.environ['LOCAL_RANK'])
        # Fallback to current device
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0

    def __init__(
        self,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        model_path: str | None = None,
        smooth_layer: int = 15,
        smooth_lr: float = 10.0,
        smooth_iter: int = 25,
        smooth_normalize_loss: bool = True,
        smooth_win: int = 3,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.smooth_layer = smooth_layer
        self.smooth_lr = smooth_lr
        self.smooth_iter = smooth_iter
        self.smooth_normalize_loss = smooth_normalize_loss
        self.smooth_win = smooth_win

        assert self.model_path is not None, "`model_path` must be provided."

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        MODEL_CLS = None

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        elif listinstr(['2.5', '2_5', 'qwen25', 'mimo', 'mm-eureka', 'vl-rethinker'], self.model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)

        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.model = MODEL_CLS.from_pretrained(
            self.model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        )
        self.model.eval()

        # Learnable tokens related
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        self.vmask = None
        
        # Initialize loss for grad mode (will be set to tensor during forward pass)
        self.loss = 0.0
        
        self.layer_indices = [self.smooth_layer]
        self._register_model_modify(self.layer_indices)

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _create_learnable_tokens(self):
        """Create learnable tokens based on actual visual token count (2x2 pooling)"""
        assert self.grid_h is not None and self.grid_w is not None, "Grid dimensions must be set"
        assert self.vmask is not None, "vmask must be set before creating learnable tokens"
        
        num_visual_tokens = self.vmask.sum().item()
        token_h = self.grid_h // 2
        token_w = self.grid_w // 2
        expected_tokens = token_h * token_w
        assert num_visual_tokens == expected_tokens, \
            f"Visual token count mismatch: expected {expected_tokens} (token_h={token_h}, token_w={token_w}), got {num_visual_tokens}"
        
        self.tokens = torch.nn.Parameter(
            torch.zeros(
                token_h, token_w, self.model.config.hidden_size,
                device=self.model.device, dtype=self.model.dtype
            )
        )
        return

    def _apply_learnable_tokens(self, hidden_states):
        """Apply learnable tokens to hidden states at visual token positions"""
        if self.tokens is None:
            return hidden_states
        
        mask_positions = torch.where(self.vmask.flatten())[0]
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_tokens = self.tokens.reshape(-1, hidden_size)
        indices = mask_positions.unsqueeze(1).expand(-1, hidden_size)
        flat_hidden_scatter_add = torch.scatter_add(flat_hidden, 0, indices, flat_tokens)
        return flat_hidden_scatter_add.view(batch_size, seq_len, hidden_size)

    def _loss_local_smooth(self, hidden_states):
        """
        Local smooth loss: within each non-overlapping smooth_win x smooth_win window,
        make visual patch features (after logit-lens mapping: norm + lm_head) similar
        by minimizing variance. Windows do not overlap.
        """
        if self.vmask is None or self.grid_h is None or self.grid_w is None:
            return hidden_states.new_tensor(0.0)
        token_h = self.grid_h // 2
        token_w = self.grid_w // 2
        n_win_h = token_h // self.smooth_win
        n_win_w = token_w // self.smooth_win
        if n_win_h == 0 or n_win_w == 0:
            return hidden_states.new_tensor(0.0)

        # Visual positions (batch 0)
        mask_positions = torch.where(self.vmask[0])[0]
        vis_h = hidden_states[0:1, mask_positions, :]
        vis_h = vis_h.squeeze(0)
        vis_h = vis_h.view(token_h, token_w, -1)

        # Logit-lens mapping: layer norm then lm_head
        ln = self.model.model.language_model.norm
        lm_head = self.model.lm_head
        vis_norm = ln(vis_h)
        vis_logit = lm_head(vis_norm)

        # Non-overlapping windows: variance within each window
        loss_list = []
        for i in range(0, token_h, self.smooth_win):
            for j in range(0, token_w, self.smooth_win):
                w = vis_logit[i : i + self.smooth_win, j : j + self.smooth_win, :]
                w = w.reshape(-1, w.shape[-1])
                if w.shape[0] <= 1:
                    continue
                mean_w = w.mean(dim=0, keepdim=True)
                var_w = ((w - mean_w) ** 2).mean()
                loss_list.append(var_w)
        if not loss_list:
            return hidden_states.new_tensor(0.0)
        return torch.stack(loss_list).sum()

    def _update_tokens(self, loss, retain_graph: bool = False):
        """Update learnable tokens using gradient descent"""
        if self.tokens is None:
            return
        
        # Skip if loss is zero or not a tensor
        if not isinstance(loss, torch.Tensor) or loss.item() == 0.0:
            return
        
        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            self.tokens,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True
        )
        
        # Update learnable tokens using gradient descent
        with torch.no_grad():
            if grads[0] is not None:
                self.tokens.data = self.tokens.data - self.smooth_lr * grads[0]

    def _zero_grad(self):
        """Zero gradients of learnable tokens"""
        if self.tokens is not None and self.tokens.grad is not None:
            self.tokens.grad.zero_()

    def _register_model_modify(self, layer_indices):
        """Register modified forward: apply learnable tokens at smooth_layer and add local smooth loss."""
        def create_model_forward(orig_forward, layer_idx):
            def model_forward(hidden_states, *args, **kwargs):
                if hidden_states.shape[1] > 1:
                    if layer_idx == self.smooth_layer:
                        h_states = self._apply_learnable_tokens(hidden_states)
                        local_loss = self._loss_local_smooth(h_states)
                        self.loss = self.loss + local_loss
                    else:
                        h_states = hidden_states
                    return orig_forward(h_states, *args, **kwargs)
                return orig_forward(hidden_states, *args, **kwargs)
            return model_forward
        
        for layer_idx in layer_indices:
            for name, module in self.model.named_modules():
                if name == f'model.language_model.layers.{layer_idx}':
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                    module.forward = create_model_forward(module._original_forward, layer_idx)
                    break

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        
        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            # Extract image path from message for concise output
            image_path = None
            question = None
            for item in message:
                if item.get('type') == 'image':
                    image_path = item.get('value', '')
                elif item.get('type') == 'text':
                    question = item.get('value', '')[:50] + "..." if len(item.get('value', '')) > 50 else item.get('value', '')
            print(f"GPU {gpu_idx} | INPUT | Image: {os.path.basename(image_path) if image_path else 'N/A'} | Q: {question or 'N/A'}")

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            _, images, _ = process_mm_info([messages], use_audio_in_video=False)
        else:
            images, _ = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        self.vmask = inputs.input_ids == self.image_token_id
        if hasattr(inputs, 'image_grid_thw'):
            assert len(inputs.image_grid_thw) == 1, f"Only single image is supported, but got {len(inputs.image_grid_thw)} images"
            grid_thw = inputs.image_grid_thw[0]
            self.grid_h = grid_thw[-2]
            self.grid_w = grid_thw[-1]
        
        self._create_learnable_tokens()
        
        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            print(f"GPU {gpu_idx} | TRAIN | Starting {self.smooth_iter} iterations")
        
        with torch.enable_grad():
            for it in range(self.smooth_iter):
                self.loss = torch.tensor(0.0, device=inputs.input_ids.device, requires_grad=True)
                self._zero_grad()
                _ = self.model(**inputs)
                if isinstance(self.loss, torch.Tensor) and self.loss.item() != 0.0:
                    if self.smooth_normalize_loss:
                        num_tokens = self.tokens.shape[0] * self.tokens.shape[1]
                        loss_for_update = self.loss / num_tokens
                    else:
                        loss_for_update = self.loss
                    retain_graph = (it < self.smooth_iter - 1)
                    self._update_tokens(loss_for_update, retain_graph=retain_graph)
                if self.verbose:
                    gpu_idx = self._get_gpu_idx()
                    if isinstance(self.loss, torch.Tensor):
                        if self.smooth_normalize_loss:
                            num_tokens = self.tokens.shape[0] * self.tokens.shape[1]
                            loss_val = (self.loss / num_tokens).item()
                        else:
                            loss_val = self.loss.item()
                    else:
                        loss_val = self.loss
                    print(f"GPU {gpu_idx} | Iter {str(it+1).zfill(len(str(self.smooth_iter)))}/{self.smooth_iter} | Loss: {loss_val:.6f}")
        
        with torch.enable_grad():
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            response_preview = response[:100] + "..." if len(response) > 100 else response
            print(f"GPU {gpu_idx} | OUTPUT | {response_preview}")
        
        # Clear learnable tokens and grid dimensions after processing
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        
        return response

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)


