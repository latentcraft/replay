import argparse
import json
import sys
from tqdm import tqdm
import os
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
import json

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
            if layer_key in captured_states:
                captured = captured_states[layer_key]

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
                return orig_forward(hidden_states, *args, **kwargs)
        return modified_forward

    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            if name == f'model.language_model.layers.{layer_idx}':
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward

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

def get_low_entropy_mask(base_entropies, rlvr_entropies, inputs, verbose=False):
    """Build mask from entropy thresholds and base vs main entropy comparison."""
    batch_size, seq_len = inputs.input_ids.shape

    total_low_entropy_mask = {layer_key: torch.zeros(batch_size, seq_len, dtype=torch.bool, device=inputs.input_ids.device) for layer_key in base_entropies.keys()}

    for layer_key in base_entropies.keys():
        base_entropy = base_entropies[layer_key]
        rlvr_entropy = rlvr_entropies[layer_key]

        low_entropy_condition = base_entropy < REPLAY_THRESHOLDS[layer_key]

        comparative_condition = base_entropy < rlvr_entropy

        layer_mask = low_entropy_condition & comparative_condition
        total_low_entropy_mask[layer_key] = total_low_entropy_mask[layer_key].to(base_entropy.device) | layer_mask.to(base_entropy.device)

    base_hook_tokens = torch.ones_like(inputs.input_ids, dtype=torch.bool)

    entropy_hook_mask.update({layer_key: base_hook_tokens.to(base_entropies[layer_key].device) & total_low_entropy_mask[layer_key].to(base_entropies[layer_key].device) for layer_key in base_entropies})

    if verbose:
        n_layers = len(base_entropies)
        b, s = base_hook_tokens.shape
        agg_both = sum(total_low_entropy_mask[lk].sum().item() for lk in total_low_entropy_mask)
        print(
            f"(debug): replay entropy mask — batch={b} seq_len={s}, {n_layers} layers; "
            f"sum_over_layers(positions with base<threshold & base<rlvr): {agg_both}"
        )

        def _layer_idx(lk: str) -> int:
            try:
                return int(lk.split("_", 1)[1])
            except (IndexError, ValueError):
                return 0

        for layer_key in sorted(base_entropies.keys(), key=_layer_idx):
            be = base_entropies[layer_key]
            re = rlvr_entropies[layer_key]
            thr = REPLAY_THRESHOLDS[layer_key]
            n_low = (be < thr).sum().item()
            n_cmp = (be < re).sum().item()
            n_both = ((be < thr) & (be < re)).sum().item()
            print(f"(debug): {layer_key} base low entropy tokens: {n_low}")
            print(f"(debug): {layer_key} base < rlvr: {n_cmp}")
            print(f"(debug): {layer_key} replay candidates (both): {n_both}")
    return

def clear():
    captured_states = {}
    entropy_hook_mask = {}
    return
#################### REPLAY ####################

def split_data(data, num_gpus):
    """
    Interleave-split data into num_gpus chunks.
    If len(values) is not divisible by num_gpus, later chunks may be smaller.
    If input is a dict, each chunk is returned as a dict.
    """
    is_dict = isinstance(data, dict)

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

    chunks = [[] for _ in range(num_gpus)]

    for i, value in enumerate(values):
        gpu_idx = i % num_gpus
        if keys is not None:
            chunks[gpu_idx].append((keys[i], value))
        else:
            chunks[gpu_idx].append(value)

    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]
    
    return chunks

VIDEO_INFO_CACHE = {}

# Prompt template
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# Question type 
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')

    parser.add_argument(
        "--model-path",
        type=str,
        default="Video-R1/Video-R1-7B",
        help="Hugging Face Hub id (or local path) for the RL Video-R1-7B checkpoint.",
    )
    default_base = os.environ.get(
        "VIDEO_R1_REPLAY_BASE_MODEL",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=default_base,
        help=(
            "Hugging Face Hub id (or local path) for foundation Qwen2.5-VL weights used in replay entropy capture. "
            "Override default with env VIDEO_R1_REPLAY_BASE_MODEL."
        ),
    )
    default_processor = os.environ.get(
        "VIDEO_R1_PROCESSOR_MODEL",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--processor-model-path",
        type=str,
        default=default_processor,
        help=(
            "HF id or local path for AutoProcessor and tokenizer. "
            "Video-R1-7B checkpoints on Hub often lack preprocessor_config (image_processor_type); "
            "default uses Qwen2.5-VL-Instruct. Override with env VIDEO_R1_PROCESSOR_MODEL."
        ),
    )
    parser.add_argument("--output-prefix", type=str, default="/path/to/out_dir")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Debug mode: at most 4 multiple-choice samples, single GPU; use --chunk_idx 0 only. Omit for full evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print (debug): replay entropy mask stats per forward (noisy).",
    )

    return parser.parse_args()

# REPLAY
def inference(
    video_path,
    question,
    base,
    model,
    processor,
    max_new_tokens=2048,
    device="cuda",
    pred_glue=None,
    problem_type="multiple choice",
    replay_verbose=False,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    'key_time': None,
                    "max_pixels": 200704, # max pixels for each frame
                    "nframes": 16  # max frame number
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[problem_type]
                },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    #################### REPLAY ####################
    with torch.no_grad():
        base_entropies = compute_base_entropies(base, inputs, list(range(REPLAY_LAYERS)))
        rlvr_entropies = compute_rlvr_entropies(model, inputs, list(range(REPLAY_LAYERS)))
        get_low_entropy_mask(base_entropies, rlvr_entropies, inputs, verbose=replay_verbose)

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        clear() # captured_states = {}; low_entropy_mask = {}
    #################### REPLAY ####################

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


def create_work_items(data, video_root):
    examples = []
    for i, info in enumerate(data):
        video_path = os.path.join(video_root, '/'.join(info['video'].split('/')[-2:]))

        example = {
            "problem": {
                "question": info['question'], 
                "options": info['choices']
            },
            "solution": {
                "answer": info['answer']
            },
            "video_path": video_path,
            "question_type" : info['question_type'],
            "video_id": i,
        }

        examples.append(example)
    return examples


#################### REPLAY ####################
def setup_model(model_path, base_model_path, processor_model_path, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(processor_model_path)
    tokenizer = AutoTokenizer.from_pretrained(processor_model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    # Replay: capture hooks on base, modify hooks on RL model
    register_cap_hook(
        base,
        list(range(0, REPLAY_LAYERS))
    )
    register_mod_hook(
        model,
        list(range(0, REPLAY_LAYERS))
    )
    return base, model, processor
#################### REPLAY ####################


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

    matches = re.search(r"[ABCDEFGHIJKLMN]", s)
    if matches is None:
        return ""
    return matches[0]


def append_to_jsonl(file_path, data):
    """Append one JSON object as a single line to a JSONL file."""
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Error writing JSONL: {e}")


def process_work_items(
    work_items,
    model_path,
    base_model_path,
    processor_model_path,
    device,
    output_prefix,
    replay_verbose=False,
):
    base, model, processor = setup_model(model_path, base_model_path, processor_model_path, device)

    log_path = f"{output_prefix}_{device}.jsonl"
    # print(log_path)
    pbar = tqdm(work_items)
    
    for _, item in enumerate(pbar):
        answers = []
        accs = []
        
        video_path = item['video_path']
        example_prompt = item["problem"]["question"]

        option = ''
        for ii, op in item["problem"]["options"].items():
            option += ii + '.' + op + '\n'
        
        prompt = example_prompt + option
        
        ans = inference(
            video_path,
            prompt,
            base,
            model,
            processor,
            device=device,
            pred_glue=None,
            problem_type="multiple choice",
            replay_verbose=replay_verbose,
        )
        print(ans)
        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, ans, re.DOTALL)

        acc = 0.0
        converted_answer = item['solution']["answer"]

        if match_answer:
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(converted_answer):
                acc = 1.0

        accs.append(acc)

        # IoU
        answers.append(ans)

        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':item["solution"], 'pred':answers, 'acc':accs }
        append_to_jsonl(log_path, item_res)

    del model, processor
    return accs

def evaluate(data, video_root, slurm_procid, args):
    work_items = create_work_items(data, video_root=video_root)

    accs = process_work_items(
        work_items,
        args.model_path,
        args.base_model_path,
        args.processor_model_path,
        "cuda",
        f"{args.output_prefix}_{slurm_procid}",
        replay_verbose=args.verbose,
    )

    return accs



if __name__=='__main__':
    args = get_args()

    num_gpus_full = 8
    gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {gpu_count}")

    path = 'replay_eval/jsons/mmvu_val.json'
    root = 'Evaluation/MMVU'

    with open(path, 'r') as f:
        data = json.load(f)

    data = [item for item in data if item['question_type'] == "multiple choice"]

    if args.dev:
        data = data[:4]
        num_gpus = 1
        print(f"DEV mode: running inference on {len(data)} sample(s) (cap 4). Use --chunk_idx 0 only.")
        if args.chunk_idx != 0:
            print(f"DEV mode: chunk_idx={args.chunk_idx} has no work; exiting.")
            sys.exit(0)
    else:
        num_gpus = num_gpus_full

    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[args.chunk_idx]

    acc = evaluate(current_data_chunk, root, args.chunk_idx, args)
    print(acc)