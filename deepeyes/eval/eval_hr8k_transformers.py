# https://github.com/Visual-Agent/DeepEyes/issues/60
import argparse
import json
import os
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
    将数据交错分割为 num_gpus 块。
    如果数据量不能被 num_gpus 整除，后面的块会包含较少元素。
    如果数据是字典，则返回的每个块也是字典。
    """
    # 记录原始数据类型
    is_dict = isinstance(data, dict)

    # 转换为可索引结构
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
    
    # 初始化空列表
    chunks = [[] for _ in range(num_gpus)]

    # 交错填充
    for i, value in enumerate(values):
        gpu_idx = i % num_gpus
        if keys is not None:  # 如果是字典，保存 key-value 对
            chunks[gpu_idx].append((keys[i], value))
        else:  # 否则只保存值
            chunks[gpu_idx].append(value)

    # 如果原始数据是字典，将每个块转换回字典
    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]
    
    return chunks

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for visual guided search on VStar')
    
    parser.add_argument("--model_base", type=str, default="/path/to/model")
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/out_dir")
    parser.add_argument("--chunk_idx", type=int, default=0)
    
    return parser.parse_args()


def inference(image, pil_image, prompt, answer, model, processor, max_new_tokens=8192, device="cuda", pred_glue=None):
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

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

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
        print(f"Error!!!!", e)
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

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
    return model, processor


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
    追加模式写入 JSONL 文件。

    参数:
        file_path (str): JSONL 文件路径。
        data (dict): 要写入的 JSON 对象（Python 字典）。
    """
    try:
        # 以追加模式打开文件
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将数据序列化为 JSON 字符串并写入文件
            json_line = json.dumps(data, ensure_ascii=False)  # 确保非 ASCII 字符正确编码
            f.write(json_line + '\n')  # 每行一个 JSON 对象
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def process_work_items(work_items, model_base, device, checkpoint_dir):
    model, processor = setup_model(model_base, device)

    log_path = f"{checkpoint_dir}_{device}.jsonl"
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
        
        ans = inference(image, pil_image, prompt, item['answer'], model, processor, device=device, pred_glue=pred_glue)
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
        args.model_base, 
        'cuda', 
        f'{args.checkpoint_dir}_{slurm_procid}',
    )

    return accs

if __name__=='__main__':
    args = get_args()

    num_gpus = 4
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    
    root = 'Evaluation/HR-Bench'
    path = os.path.join(root, 'hr_bench_8k' + '.tsv')

    df = pd.read_csv(path, sep='\t')
    data = df.to_dict(orient='records')

    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[args.chunk_idx]

    acc = evaluate(current_data_chunk, args.chunk_idx, args)
    print(acc)