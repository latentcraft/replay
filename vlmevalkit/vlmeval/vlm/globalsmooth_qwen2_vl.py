from __future__ import annotations

import os
import os.path as osp
import torch
import torch.nn.functional as F
import logging
import warnings
from PIL import Image

from torchvision.transforms import ColorJitter

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from .qwen2_vl.model import ensure_image_url
from ..smp import get_gpu_memory, listinstr


class GlobalSmoothQwen2VLChat(Qwen2VLPromptMixin, BaseModel):
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
        replay_layers: int = 16,
        replay_lr: float = 10.0,
        replay_iter: int = 25,
        replay_normalize_loss: bool = True,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        # replay-specific settings (used by your customized logic)
        self.model_path = model_path
        self.replay_layers = replay_layers
        self.replay_lr = replay_lr
        self.replay_iter = replay_iter
        self.replay_normalize_loss = replay_normalize_loss

        assert self.model_path is not None, "`model_path` must be provided (teac and stud both load from it)."

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

        # Get image token id from processor
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # Teacher and student both from model_path (same model_path)
        self.teac = MODEL_CLS.from_pretrained(
            self.model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        )
        self.teac.eval()
        self.stud = MODEL_CLS.from_pretrained(
            self.model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        )
        self.stud.eval()

        # Teacher hidden states (target for global smooth loss)
        self.teac_token_state = {}
        
        # Initialize learnable tokens related variables
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        self.vmask = None
        
        # Initialize loss for grad mode (will be set to tensor during forward pass)
        self.loss = 0.0
        
        # Register hooks on student (learnables on stud input)
        if isinstance(self.replay_layers, int):
            self.layer_indices = list(range(self.replay_layers))
        else:
            self.layer_indices = self.replay_layers
        
        self._register_stud_modify(self.layer_indices)

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

    def _apply_weak_color_aug(self, images):
        """Apply random weak color augmentation for student input (teac gets original). Expects list of PIL Images."""
        # Weak: small brightness/contrast/saturation/hue jitter
        jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        )
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(jitter(img))
            else:
                out.append(img)
        return out

    def _teac_forward(self, message, dataset=None, data_id=None):
        """Run teacher (teac) to get target hidden states at specified layers."""
        assert data_id is not None, "data_id must not be None in _teac_forward"
        with torch.no_grad():
            if listinstr(['omni'], self.model_path.lower()):
                from qwen_omni_utils import process_mm_info
            else:
                from qwen_vl_utils import process_vision_info
            
            teac_messages = []
            if self.system_prompt is not None:
                teac_messages.append({'role': 'system', 'content': self.system_prompt})
            teac_content = self._prepare_content(message, dataset=dataset)
            teac_messages.append({'role': 'user', 'content': teac_content})
            
            teac_text = self.processor.apply_chat_template([teac_messages], tokenize=False, add_generation_prompt=True)
            if listinstr(['omni'], self.model_path.lower()):
                _, teac_images, _ = process_mm_info([teac_messages], use_audio_in_video=False)
            else:
                teac_images, _ = process_vision_info([teac_messages])
            teac_inputs = self.processor(text=teac_text, images=teac_images, padding=True, return_tensors='pt')
            teac_inputs = teac_inputs.to('cuda')
            teac_generated = self.teac(
                **teac_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            
            for layer_idx in self.layer_indices:
                if layer_idx < len(teac_generated.hidden_states) - 1:
                    hidden_state = teac_generated.hidden_states[layer_idx + 1]
                    layer_key = f"layer_{str(layer_idx).zfill(2)}"
                    self.teac_token_state[layer_key] = hidden_state.detach().clone()
        return
    
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
                token_h, token_w, self.stud.config.hidden_size,
                device=self.stud.device, dtype=self.stud.dtype
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

    def _clear_teac_token_state(self):
        """Clear the teac_token_state dictionary"""
        self.teac_token_state.clear()

    def _loss_kl_div(self, layer_key, loss_positions, activations, topv=50):
        """Compute KL divergence: target=teac, current=stud at selected positions"""
        teac_act = self.teac_token_state[layer_key][loss_positions].detach()
        stud_act = activations[loss_positions]
        
        n_layers = len(self.stud.language_model.layers)
        last_lkey = f"layer_{str(n_layers - 1).zfill(2)}"
        
        if layer_key != last_lkey:
            with torch.no_grad():
                teac_var = teac_act.pow(2).mean(-1, keepdim=True)
                teac_norm_h = teac_act * torch.rsqrt(
                    teac_var + self.teac.language_model.norm.variance_epsilon
                )
                teac_logits = self.teac.language_model.norm.weight.data * teac_norm_h
            
            stud_var = stud_act.pow(2).mean(-1, keepdim=True)
            stud_norm_h = stud_act * torch.rsqrt(
                stud_var + self.stud.language_model.norm.variance_epsilon
            )
            stud_logits = self.stud.language_model.norm.weight.data * stud_norm_h
        else:
            with torch.no_grad():
                teac_logits = teac_act
            stud_logits = stud_act
        
        with torch.no_grad():
            teac_lm_logits = teac_logits @ self.teac.lm_head.weight.data.T
            topv_teac_logits, topv_teac_idx = torch.topk(teac_lm_logits, k=topv, dim=-1)
            topv_teac_probs = F.softmax(topv_teac_logits, dim=-1)
        
        stud_lm_logits = stud_logits @ self.stud.lm_head.weight.data.T
        row_idx = torch.arange(topv_teac_idx.shape[0], device=topv_teac_idx.device)
        topv_stud_logits = stud_lm_logits[row_idx[:, None], topv_teac_idx]
        topv_stud_logprobs = F.log_softmax(topv_stud_logits, dim=-1)
        
        kl_loss = F.kl_div(
            topv_stud_logprobs,
            topv_teac_probs,
            reduction='batchmean'
        )
        return kl_loss

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
                self.tokens.data = self.tokens.data - self.replay_lr * grads[0]

    def _zero_grad(self):
        """Zero gradients of learnable tokens"""
        if self.tokens is not None and self.tokens.grad is not None:
            self.tokens.grad.zero_()

    def _register_stud_modify(self, layer_indices):
        """Register modified forward on student (stud): learnables on input, loss vs teac targets."""
        def create_stud_forward(orig_forward, layer_idx):
            def stud_forward(hidden_states, *args, **kwargs):
                if hidden_states.shape[1] > 1:
                    lkey = f"layer_{str(layer_idx).zfill(2)}"
                    
                    if layer_idx == 0:
                        h_states = self._apply_learnable_tokens(hidden_states)
                    else:
                        h_states = hidden_states
                    
                    if lkey in self.teac_token_state:
                        teac_h = self.teac_token_state[lkey]
                        if teac_h.device != h_states.device:
                            teac_h = teac_h.to(h_states.device)
                        
                        assert teac_h.shape == h_states.shape, (
                            f"teac and stud activation length mismatch at {lkey}: "
                            f"teac {teac_h.shape} vs stud {h_states.shape}"
                        )
                        if self.vmask.sum() > 0:
                            layer_loss = self._loss_kl_div(lkey, self.vmask, h_states)
                            self.loss = self.loss + layer_loss
                    
                    return orig_forward(h_states, *args, **kwargs)
                else:
                    return orig_forward(hidden_states, *args, **kwargs)
            return stud_forward
        
        for layer_idx in layer_indices:
            for name, module in self.stud.named_modules():
                if name == f'model.language_model.layers.{layer_idx}':
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                    module.forward = create_stud_forward(module._original_forward, layer_idx)
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
        # One batch to set vmask and grid (same layout for same resolution)
        stud_images_0 = self._apply_weak_color_aug(images)
        inputs_0 = self.processor(text=text, images=stud_images_0, padding=True, return_tensors='pt')  # noqa: E501
        inputs_0 = inputs_0.to('cuda')

        # Clear previous learnable tokens before processing new data
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        
        self.vmask = inputs_0.input_ids == self.image_token_id
        if hasattr(inputs_0, 'image_grid_thw'):
            assert len(inputs_0.image_grid_thw) == 1, f"Only single image is supported, but got {len(inputs_0.image_grid_thw)} images"
            grid_thw = inputs_0.image_grid_thw[0]
            self.grid_h = grid_thw[-2]
            self.grid_w = grid_thw[-1]
        
        data_id = os.path.split(message[0]['value'])[-1].split('.')[0].zfill(4)
        self._create_learnable_tokens()
        
        self._clear_teac_token_state()
        self._teac_forward(message, dataset=dataset, data_id=data_id)
        
        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            print(f"GPU {gpu_idx} | TRAIN | Starting {self.replay_iter} iterations")
        
        with torch.enable_grad():
            for it in range(self.replay_iter):
                # Fresh augmentation every step
                stud_images = self._apply_weak_color_aug(images)
                inputs_step = self.processor(text=text, images=stud_images, padding=True, return_tensors='pt')  # noqa: E501
                inputs_step = inputs_step.to('cuda')
                
                self.loss = torch.tensor(0.0, device=inputs_step.input_ids.device, requires_grad=True)
                self._zero_grad()
                
                _ = self.stud(
                    **inputs_step
                )
                
                # Update learnable tokens
                if isinstance(self.loss, torch.Tensor) and self.loss.item() != 0.0:
                    # Normalize loss by number of tokens if enabled
                    if self.replay_normalize_loss:
                        num_tokens = self.tokens.shape[0] * self.tokens.shape[1]
                        loss_for_update = self.loss / num_tokens
                    else:
                        loss_for_update = self.loss
                    
                    retain_graph = (it < self.replay_iter - 1)
                    self._update_tokens(loss_for_update, retain_graph=retain_graph)
                
                if self.verbose:
                    gpu_idx = self._get_gpu_idx()
                    if isinstance(self.loss, torch.Tensor):
                        if self.replay_normalize_loss:
                            num_tokens = self.tokens.shape[0] * self.tokens.shape[1]
                            loss_val = (self.loss / num_tokens).item()
                        else:
                            loss_val = self.loss.item()
                    else:
                        loss_val = self.loss
                    print(f"GPU {gpu_idx} | Iter {str(it+1).zfill(len(str(self.replay_iter)))}/{self.replay_iter} | Loss: {loss_val:.6f}")
        
        # Generate with student (use one augmented view, e.g. inputs_0)
        with torch.enable_grad():
            generated_ids = self.stud.generate(
                **inputs_0,
                **self.generate_kwargs,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_0.input_ids, generated_ids)
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


