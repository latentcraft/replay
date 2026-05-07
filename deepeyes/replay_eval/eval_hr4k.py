# https://github.com/Visual-Agent/DeepEyes/issues/60
import argparse
import json
import os
import sys
import re
import base64
import torch
import math
import pandas as pd
import io
import copy
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import StoppingCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
from tqdm import tqdm

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# Print shared-batch visual token stats once per process (first forward).
_DEBUG_VISUAL_TOKENS_ONCE = True

#################### REPLAY ####################
REPLAY_LAYERS = 7
REPLAY_THRESHOLDS = {'layer_0': 4.812, 'layer_1': 4.781, 'layer_2': 3.734, 'layer_3': 3.297, 'layer_4': 3.328, 'layer_5': 3.812, 'layer_6': 4.312, 'layer_7': 5.125, 'layer_8': 4.938, 'layer_9': 5.000, 'layer_10': 4.719, 'layer_11': 4.656, 'layer_12': 4.656, 'layer_13': 4.812}
captured_states = {}
entropy_hook_mask = {}

def register_cap_hook(model, layer_indices):
    """Register forward hooks on the given layers to capture hidden states."""
    def capture_hook(module, input, output):
        layer_idx = module.self_attn.layer_idx
        layer_key = f"layer_{layer_idx}"
        # First hidden_states: input[0]
        captured_states[layer_key] = input[0].detach().clone()
    
    # Find target layers and register hooks
    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            if name == f'model.language_model.layers.{layer_idx}':
                handle = module.register_forward_hook(capture_hook)

def register_mod_hook(model, layer_indices=None):
    def create_mod_forward(orig_forward, layer_idx):
        def modified_forward(hidden_states, *args, **kwargs):
            layer_key = f"layer_{layer_idx}"
            # Check for captured states for this layer
            if layer_key in captured_states:
                captured = captured_states[layer_key]
                
                # Keep tensors on the same device
                if captured.device != hidden_states.device:
                    captured = captured.to(hidden_states.device)
                
                if captured.shape == hidden_states.shape:
                    # Entropy-based mask (not plain hook_tokens)
                    mask = entropy_hook_mask[layer_key].unsqueeze(-1).repeat(1, 1, captured.shape[-1])
                    concated = captured.to(hidden_states.device) * mask.to(hidden_states.device) + hidden_states.to(hidden_states.device) * (~mask).to(hidden_states.device)
                    
                    return orig_forward(concated, *args, **kwargs)
                else:
                    return orig_forward(hidden_states, *args, **kwargs)
            else:
                # No captured states: pass through original hidden states
                return orig_forward(hidden_states, *args, **kwargs)
        return modified_forward
    
    # Find target layers and wrap their forward
    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            if name == f'model.language_model.layers.{layer_idx}':
                # Save original forward
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward
                
                # Replace forward with modified version
                module.forward = create_mod_forward(module._original_forward, layer_idx)
                break

def compute_base_entropies(model, inputs, layer_indices=None):
    """Per-token entropies at selected layers for the base model."""
    base_entropies = {}
    
    with torch.no_grad():
        generated = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True
        )
        for layer_idx in layer_indices:
            if layer_idx < len(generated.hidden_states) - 1:
                hidden_state = generated.hidden_states[layer_idx + 1]
                if layer_idx < len(model.model.language_model.layers) - 1:
                    logit = model.model.language_model.norm(hidden_state)
                else:
                    logit = hidden_state
                
                logit = model.lm_head(logit)
                log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                prob = torch.nn.functional.softmax(logit, dim=-1)
                entropy = torch.sum(prob * -log_prob, dim=-1)
                
                base_entropies[f"layer_{layer_idx}"] = entropy
    return base_entropies

def compute_rlvr_entropies(model, inputs, layer_indices=None):
        """Per-token entropies at selected layers for the main (RL) model."""
        rlvr_entropies = {}
        
        with torch.no_grad():
            # Temporarily remove modify hooks so entropy is computed cleanly
            disable_mod_hook(model, layer_indices)
            
            try:
                generated = model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                for layer_idx in layer_indices:
                    if layer_idx < len(generated.hidden_states) - 1:
                        hidden_state = generated.hidden_states[layer_idx + 1]
                        
                        if layer_idx < len(model.model.language_model.layers) - 1:
                            logit = model.model.language_model.norm(hidden_state)
                        else:
                            logit = hidden_state
                        
                        logit = model.lm_head(logit)
                        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                        prob = torch.nn.functional.softmax(logit, dim=-1)
                        entropy = torch.sum(prob * -log_prob, dim=-1)
                        
                        rlvr_entropies[f"layer_{layer_idx}"] = entropy
            finally:
                # Restore modify hooks
                restore_mod_hook(model, layer_indices)
        
        return rlvr_entropies

def disable_mod_hook(model, layer_indices=None):
    """Temporarily disable modify hooks."""
    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            if name == f'model.language_model.layers.{layer_idx}':
                if hasattr(module, '_original_forward'):
                    module._temp_forward = module.forward
                    module.forward = module._original_forward
                break

def restore_mod_hook(model, layer_indices=None):
    """Restore modify hooks after disable_mod_hook."""
    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            if name == f'model.language_model.layers.{layer_idx}':
                if hasattr(module, '_temp_forward'):
                    module.forward = module._temp_forward
                    delattr(module, '_temp_forward')
                break

def get_low_entropy_mask(base_entropies, rlvr_entropies, inputs, verbose: bool = False):
    """Build mask from entropy thresholds and base vs main entropy comparison."""
    batch_size, seq_len = inputs.input_ids.shape
    
    # Per-layer masks (all False initially)
    total_low_entropy_mask = {layer_key: torch.zeros(batch_size, seq_len, dtype=torch.bool, device=inputs.input_ids.device) for layer_key in base_entropies.keys()}
    
    # Per-layer entropy conditions
    for layer_key in base_entropies.keys():
        base_entropy = base_entropies[layer_key]
        rlvr_entropy = rlvr_entropies[layer_key]
        
        # Condition 1: base entropy below layer threshold
        low_entropy_condition = base_entropy < REPLAY_THRESHOLDS[layer_key]
        
        # Condition 2: base entropy below main model entropy at same positions
        comparative_condition = base_entropy < rlvr_entropy
        
        # Both must hold for a position to be replay-candidate
        layer_mask = low_entropy_condition & comparative_condition
        total_low_entropy_mask[layer_key] = total_low_entropy_mask[layer_key].to(base_entropy.device) | layer_mask.to(base_entropy.device)
    
    # Combine with all-token hook template (ones)
    base_hook_tokens = torch.ones_like(inputs.input_ids, dtype=torch.bool)
    
    # Final mask written to entropy_hook_mask
    entropy_hook_mask.update({layer_key: base_hook_tokens.to(base_entropies[layer_key].device) & total_low_entropy_mask[layer_key].to(base_entropies[layer_key].device) for layer_key in base_entropies})
    
    if verbose:
        total_tokens = base_hook_tokens.sum().item() * len(base_entropies)
        low_entropy_tokens = sum([total_low_entropy_mask[layer_key].sum().item() for layer_key in total_low_entropy_mask]) 
        print(f"Total candidate tokens: {total_tokens}")
        print(f"Low entropy + comparative tokens: {low_entropy_tokens}")
        
        # Extra per-layer stats
        for layer_key in base_entropies.keys():
            base_entropy = base_entropies[layer_key]
            rlvr_entropy = rlvr_entropies[layer_key]
            low_entropy_count = (base_entropy < REPLAY_THRESHOLDS[layer_key]).sum().item()
            comparative_count = (base_entropy < rlvr_entropy).sum().item()
            both_conditions_count = ((base_entropy < REPLAY_THRESHOLDS[layer_key]) & 
                                (base_entropy < rlvr_entropy)).sum().item()
            print(f"{layer_key} - Low entropy: {low_entropy_count}, Hook<Model: {comparative_count}, Both: {both_conditions_count}")
    return

def clear():
    captured_states = {}
    entropy_hook_mask = {}
    return
#################### REPLAY ####################

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

USER_PROMPT = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_init = """Question: {question}
Options: {options}
""" + USER_PROMPT

user_prompt = USER_PROMPT

start_token = "<tool_call>"
end_token = "</tool_call>"

def split_data(data, num_gpus):
    """
    Interleave-split data into num_gpus chunks.
    If len(values) is not divisible by num_gpus, later chunks may be smaller.
    If input is a dict, each chunk is returned as a dict.
    """
    # Remember whether input was a dict
    is_dict = isinstance(data, dict)

    # Normalize to indexable values (and keys if dict)
    if is_dict:
        keys = list(data.keys())
        values = list(data.values())
    elif isinstance(data, list):
        keys = None
        values = data
    else:
        values = list(data)
        keys = None

    data_size = len(values)
    
    # One list per GPU chunk
    chunks = [[] for _ in range(num_gpus)]

    # Round-robin assignment
    for i, value in enumerate(values):
        gpu_idx = i % num_gpus
        if keys is not None:  # dict: keep (key, value) pairs
            chunks[gpu_idx].append((keys[i], value))
        else:
            chunks[gpu_idx].append(value)

    # Rebuild dict chunks if needed
    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]
    
    return chunks

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for visual guided search on VStar')
    
    parser.add_argument("--model-path", type=str, default="/path/to/model")
    default_base = os.environ.get(
        "DEEPEYES_REPLAY_BASE_MODEL",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=default_base,
        help=(
            "Foundation (pre-RL) Qwen2.5-VL weights for replay entropy capture. "
            "Processor is always loaded from --model-path so inputs match the RL model. "
            "Override default with env DEEPEYES_REPLAY_BASE_MODEL. "
            "If forward fails (e.g. vocab/embedding mismatch), use a base checkpoint aligned with DeepEyes tokenizer."
        ),
    )
    parser.add_argument("--output-prefix", type=str, default="/path/to/out_dir")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: at most 4 samples, single chunk (use --chunk_idx 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print [debug] diagnostics (visual token counts, tensor shapes).",
    )

    return parser.parse_args()


def print_debug_visual_token_count(processor, inputs, verbose: bool = False):
    """Qwen2.5-VL: count vision placeholder tokens in input_ids (one batch for both base and rlvr)."""
    if not verbose:
        return
    ids = inputs["input_ids"]
    tok = processor.tokenizer

    def count_special(name):
        tid = tok.convert_tokens_to_ids(name)
        if tid is None or (tok.unk_token_id is not None and tid == tok.unk_token_id):
            return None
        return int((ids == tid).sum().item())

    n_img = count_special("<|image_pad|>")
    n_vid = count_special("<|video_pad|>")
    parts = []
    if n_img is not None:
        parts.append(f"<|image_pad|>={n_img}")
    if n_vid is not None and n_vid > 0:
        parts.append(f"<|video_pad|>={n_vid}")
    detail = ", ".join(parts) if parts else "(no <|image_pad|>/<|video_pad|> counted; check tokenizer)"

    n_vis = n_img if n_img is not None else (n_vid if n_vid is not None else -1)
    print(f"[debug] visual placeholders in batch (shared by base & rlvr): {detail}")
    print(f"[debug] base vs rlvr (same `inputs`): visual pad token total = {n_vis} vs {n_vis} (expect equal)")

    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        print(f"[debug] image_grid_thw: {inputs['image_grid_thw'].detach().cpu().tolist()}")
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        pv = inputs["pixel_values"]
        print(f"[debug] pixel_values shape: {tuple(pv.shape)}")


def inference(
    image,
    pil_image,
    prompt,
    answer,
    base,
    model,
    processor,
    max_new_tokens=8192,
    device="cuda",
    pred_glue=None,
    verbose: bool = False,
):
    global _DEBUG_VISUAL_TOKENS_ONCE
    base64_image = encode_pil_image_to_base64(image)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "mix_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64",
                    "mix_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    chat_message = messages
    response_message = ""

    # https://github.com/Visual-Agent/DeepEyes/issues/16
    stop_words = ["<|im_end|>\n".strip(), "</tool_call>"]
    stop_token_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in stop_words]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    
    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break
            if try_count > 10:
                break
            text = processor.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)

            image_inputs, _ = process_vision_info(chat_message)
            generation_kwargs = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "stopping_criteria": stopping_criteria,
            }

            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            if _DEBUG_VISUAL_TOKENS_ONCE and try_count == 0 and turn_idx == 0:
                if verbose:
                    print_debug_visual_token_count(processor, inputs, verbose=True)
                _DEBUG_VISUAL_TOKENS_ONCE = False

            with torch.no_grad():
                base_entropies = compute_base_entropies(base, inputs, list(range(REPLAY_LAYERS)))
                rlvr_entropies = compute_rlvr_entropies(model, inputs, list(range(REPLAY_LAYERS)))
                get_low_entropy_mask(base_entropies, rlvr_entropies, inputs, verbose=verbose)
                
                output_ids = model.generate(**inputs, **generation_kwargs)
                clear() # captured_states = {}; low_entropy_mask = {}

            generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
            response_message = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print(f'\033[92m{response_message}\033[0m')
    
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list['arguments']['bbox_2d']
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_image.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)

                cropped_pil_image_content = {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64, {cropped_pil_image}",
                    "mix_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS
                }
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []

                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                chat_message.extend(_message)
            
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url", 
                                "image_url": f"data:image/jpeg;base64,"
                            },
                            {
                                "type": "text", 
                                "text": user_msg
                            },
                        ],
                    }
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)
            
            try_count += 1
    except Exception as e:
        print(f"Error: {e}")
        status = 'error'
    
    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['question'] = prompt
    save_info['answer'] = answer
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['status'] = status
    return save_info


def create_work_items(data):
    examples = []
    for i, info in enumerate(data):
        example = info
        examples.append(example)
    return examples

def setup_model(model_path, base_model_path, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    # Single processor from the RL model — base and main share identical preprocessing.
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Replay: register capture hooks on base model and modify hooks on main model
    register_cap_hook(
        base,
        list(range(0, REPLAY_LAYERS))
    )
    register_mod_hook(
        model,
        list(range(0, REPLAY_LAYERS))
    )
    return base, model, processor


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


def append_to_jsonl(file_path, data):
    """
    Append one JSON object as a single line to a JSONL file.

    Args:
        file_path: Path to the JSONL file.
        data: Dict to serialize.
    """
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)  # preserve non-ASCII
            f.write(json_line + '\n')  # one JSON object per line
    except Exception as e:
        print(f"Error writing JSONL: {e}")

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def process_work_items(work_items, model_path, base_model_path, device, output_prefix, verbose: bool = False):
    base, model, processor = setup_model(model_path, base_model_path, device)

    log_path = f"{output_prefix}_{device}.jsonl"
    pbar = tqdm(work_items)
    
    for idx, item in enumerate(pbar):
        pred_glue = None
        answers = []
        accs = []
        previous = []
            
        image = item['image']
        image = decode_base64_to_image(image)
        ori_image = copy.deepcopy(image)
        ori_width, ori_height = image.size
        category = item['category']
        pil_image = ori_image.copy()

        resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
        image = image.resize((resize_w, resize_h), resample=Image.BICUBIC)

        question = item['question']
        options = ['A. ' + item['A'], 'B. ' + item['B'], 'C. ' + item['C'], 'D. ' + item['D']]

        option_str = "\n"
        for i in range(len(options)):
            option_str += options[i] + '\n'

        prompt = instruction_prompt_init.format(question=question, options=option_str)
        
        ans = inference(
            image,
            pil_image,
            prompt,
            item['answer'],
            base,
            model,
            processor,
            device=device,
            pred_glue=pred_glue,
            verbose=verbose,
        )
        print(ans['pred_ans'])
        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, ans['pred_ans'], re.DOTALL)

        acc = 0.0
        converted_answer = item["answer"]

        if match_answer:
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(converted_answer):
                acc = 1.0
        else:
            answer = ans['pred_ans']
            if extract_characters_regex(answer) == extract_characters_regex(converted_answer):
                acc = 1.0

        accs.append(acc)
        answers.append(ans['pred_ans'])
        
        pattern_sum = r'<think>(.*?)</think>'
        match_sum = re.search(pattern_sum, ans['pred_ans'], re.DOTALL)
        if match_sum:
            previous.append(match_sum.group(1))
        
        pred_glue = None

        item_res = {'prompt': prompt, 'gt': item["answer"], 'pred': answers, 'category': category, 'acc': accs}
        append_to_jsonl(log_path, item_res)

    del model, processor
    return accs

def evaluate(data, slurm_procid, args):
    work_items = create_work_items(data)

    accs = process_work_items(
        work_items,
        args.model_path,
        args.base_model_path,
        'cuda',
        f'{args.output_prefix}_{slurm_procid}',
        verbose=args.verbose,
    )

    return accs

if __name__=='__main__':
    args = get_args()

    gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {gpu_count}")

    root = 'Evaluation/HR-Bench'
    path = os.path.join(root, 'hr_bench_4k' + '.tsv')

    df = pd.read_csv(path, sep='\t')
    data = df.to_dict(orient='records')

    if args.dev:
        data = data[:4]
        print(f"DEV mode: running inference on {len(data)} sample(s) (cap 4). Use --chunk_idx 0 only.")
        num_gpus = 1
        data_chunks = split_data(data, num_gpus)
        if args.chunk_idx != 0:
            print(f"DEV mode: chunk_idx={args.chunk_idx} has no work; exiting.")
            sys.exit(0)
        current_data_chunk = data_chunks[0]
    else:
        num_gpus = 4
        data_chunks = split_data(data, num_gpus)
        current_data_chunk = data_chunks[args.chunk_idx]

    acc = evaluate(current_data_chunk, args.chunk_idx, args)
    print(acc)