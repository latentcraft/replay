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


class MidReplayQwen2VLChat(Qwen2VLPromptMixin, BaseModel):
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
        rlvr_path: str | None = None,
        base_path: str | None = None,
        replay_layer: int = 15,
        replay_entropy_threshold: float = 0.4,
        replay_lr: float = 10.0,
        replay_iter: int = 25,
        replay_save: str | None = None,
        replay_normalize_loss: bool = True,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        # replay-specific settings (used by your customized logic)
        self.rlvr_path = rlvr_path
        self.base_path = base_path
        self.replay_layer = replay_layer
        self.replay_entropy_threshold = replay_entropy_threshold
        self.replay_lr = replay_lr
        self.replay_iter = replay_iter
        self.replay_save = replay_save
        self.replay_normalize_loss = replay_normalize_loss

        # NOTE:
        # - `rlvr_path` is the actual model path to load (it was previously
        #   passed as `model_path` in older versions).
        # - `base_path` is kept for your replay logic (e.g. base weights),
        #   but is NOT used as the primary load target here.
        assert self.rlvr_path is not None, "`rlvr_path` must be provided."
        assert self.base_path is not None, "`base_path` must be provided."
        assert self.replay_save is not None, "`replay_save` must be provided."

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

        if listinstr(['omni'], self.rlvr_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.rlvr_path)
        elif listinstr(['2.5', '2_5', 'qwen25', 'mimo', 'mm-eureka', 'vl-rethinker'], self.rlvr_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.rlvr_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(self.rlvr_path)

        # Get image token id from processor
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.rlvr = MODEL_CLS.from_pretrained(
            self.rlvr_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        )
        self.rlvr.eval()

        # Load base model
        BASE_MODEL_CLS = None
        if listinstr(['omni'], self.base_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            BASE_MODEL_CLS = Qwen2_5OmniForConditionalGeneration
        elif listinstr(['2.5', '2_5', 'qwen25', 'mimo'], self.base_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration
            BASE_MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration
            BASE_MODEL_CLS = Qwen2VLForConditionalGeneration

        self.base = BASE_MODEL_CLS.from_pretrained(
            self.base_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        )
        self.base.eval()

        # Initialize base model state dictionaries
        self.base_token_entropy = {}
        self.base_token_state = {}
        
        # Initialize learnable tokens related variables
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        self.vmask = None
        
        # Initialize loss for grad mode (will be set to tensor during forward pass)
        self.loss = 0.0
        
        # Register hooks for rlvr model (single layer only)
        self.layer_indices = [self.replay_layer]
        self._register_rlvr_modify(self.layer_indices)

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

    def _clear_base_token_entropy(self):
        """Clear the base_token_entropy dictionary"""
        self.base_token_entropy.clear()

    def _base_forward(self, message, dataset=None, data_id=None):
        """Compute token entropies and hidden states of the base model at specified layers"""
        assert data_id is not None, "data_id must not be None in _base_forward"
        with torch.no_grad():
            # Generate inputs for base model from message, with optional image resizing
            if listinstr(['omni'], self.rlvr_path.lower()):
                from qwen_omni_utils import process_mm_info
            else:
                from qwen_vl_utils import process_vision_info
            
            # Use same content as rlvr path (including min_pixels/max_pixels/total_pixels)
            # so base and rlvr get identical image resolution and visual token count.
            base_messages = []
            if self.system_prompt is not None:
                base_messages.append({'role': 'system', 'content': self.system_prompt})
            base_content = self._prepare_content(message, dataset=dataset)
            base_messages.append({'role': 'user', 'content': base_content})
            
            base_text = self.processor.apply_chat_template([base_messages], tokenize=False, add_generation_prompt=True)
            if listinstr(['omni'], self.rlvr_path.lower()):
                _, base_images, _ = process_mm_info([base_messages], use_audio_in_video=False)
            else:
                base_images, _ = process_vision_info([base_messages])
            base_inputs = self.processor(text=base_text, images=base_images, padding=True, return_tensors='pt')
            base_inputs = base_inputs.to('cuda')
            self._debug_base_input_len = base_inputs.input_ids.shape[1]
            base_visual = (base_inputs.input_ids == self.image_token_id).sum().item()
            self._debug_base_visual_count = base_visual
            print(f"[DEBUG] base_inputs.input_ids.shape={base_inputs.input_ids.shape} len={self._debug_base_input_len} visual_tokens={base_visual}")
            
            base_generated = self.base(
                **base_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            
            for layer_idx in self.layer_indices:
                if layer_idx < len(base_generated.hidden_states) - 1:
                    hidden_state = base_generated.hidden_states[layer_idx + 1]
                    layer_key = f"layer_{str(layer_idx).zfill(2)}"
                    self.base_token_state[layer_key] = hidden_state.detach().clone()
                    
                    # Compute entropy from upsampled hidden state
                    if layer_idx < len(self.base.language_model.layers) - 1:
                        logit = self.base.language_model.norm(hidden_state)
                    else:
                        logit = hidden_state
                    
                    logit = self.base.lm_head(logit)
                    log_prob = F.log_softmax(logit, dim=-1)
                    prob = F.softmax(logit, dim=-1)
                    entropy = torch.sum(prob * -log_prob, dim=-1)
                    
                    self.base_token_entropy[layer_key] = entropy
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
                token_h, token_w, self.rlvr.config.hidden_size,
                device=self.rlvr.device, dtype=self.rlvr.dtype
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

    def _clear_base_token_state(self):
        """Clear base token state dictionary"""
        self.base_token_state.clear()

    def _loss_kl_div(self, layer_key, loss_positions, activations, topv=50):
        """Compute KL divergence between base and rlvr activations at selected positions"""
        base_act = self.base_token_state[layer_key][loss_positions].detach()
        rlvr_act = activations[loss_positions]
        
        n_layers = len(self.rlvr.language_model.layers)
        last_lkey = f"layer_{str(n_layers - 1).zfill(2)}"
        
        if layer_key != last_lkey:
            # ** base logits ** #
            with torch.no_grad():
                base_var = base_act.pow(2).mean(-1, keepdim=True)
                base_norm_h = base_act * torch.rsqrt(
                    base_var + self.base.language_model.norm.variance_epsilon
                )
                base_logits = self.base.language_model.norm.weight.data * base_norm_h
            
            # ** rlvr logits ** #
            rlvr_var = rlvr_act.pow(2).mean(-1, keepdim=True)
            rlvr_norm_h = rlvr_act * torch.rsqrt(
                rlvr_var + self.rlvr.language_model.norm.variance_epsilon
            )
            rlvr_logits = self.rlvr.language_model.norm.weight.data * rlvr_norm_h
        else:
            with torch.no_grad():
                base_logits = base_act
            rlvr_logits = rlvr_act
        
        # Compute logits and probabilities
        with torch.no_grad():
            base_lm_logits = base_logits @ self.base.lm_head.weight.data.T
            topv_base_logits, topv_base_idx = torch.topk(base_lm_logits, k=topv, dim=-1)
            topv_base_probs = F.softmax(topv_base_logits, dim=-1)
        
        rlvr_lm_logits = rlvr_logits @ self.rlvr.lm_head.weight.data.T
        row_idx = torch.arange(topv_base_idx.shape[0], device=topv_base_idx.device)
        topv_rlvr_logits = rlvr_lm_logits[row_idx[:, None], topv_base_idx]
        topv_rlvr_logprobs = F.log_softmax(topv_rlvr_logits, dim=-1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            topv_rlvr_logprobs,
            topv_base_probs,
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

    def _register_rlvr_modify(self, layer_indices):
        """Register modified forward on rlvr model to apply learnable tokens and compute loss"""
        def create_rlvr_forward(orig_forward, layer_idx):
            def rlvr_forward(hidden_states, *args, **kwargs):
                if hidden_states.shape[1] > 1:
                    lkey = f"layer_{str(layer_idx).zfill(2)}"
                    
                    # Apply learnable tokens at replay_layer (mid representation)
                    if layer_idx == self.replay_layer:
                        h_states = self._apply_learnable_tokens(hidden_states)
                    else:
                        h_states = hidden_states
                    
                    # Compute loss
                    if lkey in self.base_token_state:
                        base_h = self.base_token_state[lkey]
                        if base_h.device != h_states.device:
                            base_h = base_h.to(h_states.device)
                        
                        assert base_h.shape == h_states.shape, (
                            f"base and rlvr activation length mismatch at {lkey}: "
                            f"base {base_h.shape} vs rlvr {h_states.shape}"
                        )
                        # Compute low entropy condition
                        mask_entropy = self.base_token_entropy[lkey].clone()
                        mask_entropy[~self.vmask] = float('inf')
                        
                        # Low entropy condition: tokens with entropy <= threshold * max_entropy
                        kth_val = mask_entropy[self.vmask.cpu()].max() * self.replay_entropy_threshold
                        low_ent_mask = mask_entropy <= kth_val
                        
                        # Compute loss if there are low entropy tokens
                        if low_ent_mask.sum() > 0:
                            layer_loss = self._loss_kl_div(lkey, low_ent_mask, h_states)
                            self.loss = self.loss + layer_loss
                    
                    return orig_forward(h_states, *args, **kwargs)
                else:
                    return orig_forward(hidden_states, *args, **kwargs)
            return rlvr_forward
        
        # Find target layers and replace forward method
        for layer_idx in layer_indices:
            for name, module in self.rlvr.named_modules():
                if name == f'model.language_model.layers.{layer_idx}':
                    # Save original forward
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward
                    
                    # Replace forward method
                    module.forward = create_rlvr_forward(module._original_forward, layer_idx)
                    break

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.rlvr_path.lower()):
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
        if listinstr(['omni'], self.rlvr_path.lower()):
            _, images, _ = process_mm_info([messages], use_audio_in_video=False)
            inputs = self.processor(text=text, images=images, padding=True, return_tensors='pt')  # noqa: E501
        else:
            images, _ = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')
        rlvr_visual = (inputs.input_ids == self.image_token_id).sum().item()
        print(f"[DEBUG] inputs.input_ids.shape={inputs.input_ids.shape} len={inputs.input_ids.shape[1]} visual_tokens={rlvr_visual}")

        # Clear previous learnable tokens before processing new data
        self.tokens = None
        self.grid_h = None
        self.grid_w = None
        
        # Set vmask and grid dimensions for current input
        self.vmask = inputs.input_ids == self.image_token_id
        
        # Assert single image only
        if hasattr(inputs, 'image_grid_thw'):
            assert len(inputs.image_grid_thw) == 1, f"Only single image is supported, but got {len(inputs.image_grid_thw)} images"
            grid_thw = inputs.image_grid_thw[0]
            self.grid_h = grid_thw[-2]
            self.grid_w = grid_thw[-1]
        
        # Create learnable tokens and run gradient updates
        data_id = os.path.split(message[0]['value'])[-1].split('.')[0].zfill(4)
        self._create_learnable_tokens()
        
        # Clear previous state and compute base model entropies
        self._clear_base_token_state()
        self._clear_base_token_entropy()
        self._base_forward(message, dataset=dataset, data_id=data_id)
        base_len = getattr(self, '_debug_base_input_len', None)
        base_visual = getattr(self, '_debug_base_visual_count', None)
        rlvr_visual = self.vmask.sum().item()
        print(f"[DEBUG] input_ids length: base={base_len} rlvr={inputs.input_ids.shape[1]} same={base_len == inputs.input_ids.shape[1] if base_len is not None else 'N/A'}")
        print(f"[DEBUG] visual_tokens: base={base_visual} rlvr={rlvr_visual} same={base_visual == rlvr_visual if base_visual is not None else 'N/A'}")
        
        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            print(f"GPU {gpu_idx} | TRAIN | Starting {self.replay_iter} iterations")
        
        with torch.enable_grad():
            for it in range(self.replay_iter):
                # Initialize loss as tensor 0.0 for gradient tracking
                self.loss = torch.tensor(0.0, device=inputs.input_ids.device, requires_grad=True)
                self._zero_grad()
                
                # Forward pass to compute loss
                _ = self.rlvr(
                    **inputs
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
        
        # Save learnable tokens after iteration
        if os.path.isfile(self.replay_save):
            token_path = self.replay_save
        else:
            data_id = os.path.split(message[0]['value'])[-1].split('.')[0].zfill(4)
            token_path = osp.join(self.replay_save, dataset, f"{data_id}.pt") if dataset else osp.join(self.replay_save, f"{data_id}.pt")
            os.makedirs(osp.dirname(token_path), exist_ok=True)
        
        if self.verbose:
            gpu_idx = self._get_gpu_idx()
            print(f"GPU {gpu_idx} | SAVE | Saving tokens to: {os.path.basename(token_path)}")
        torch.save(self.tokens, token_path)
        
        # Generate with learnable tokens
        with torch.enable_grad():
            generated_ids = self.rlvr.generate(
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


