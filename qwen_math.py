from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys
import time
from latex2sympy2 import latex2sympy
from PIL import Image

def log(msg, *args, **kwargs):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    print(f"{timestamp} {msg}", flush=True, *args, **kwargs)
    
def recheck_answer(s, tag):
    if tag == "选择题":
        if s not in ["A", "B", "C", "D", "E", "F"]:
            return "B"
        else:
            return s
    else:
        try:
            num = float(s)
            return "{:.6f}".format(num)
        except ValueError:
            return "1.000000"    

def format_string_value(s):
    # 检查输入是否为 None
    if s is None:
        return ""  # 或者可以根据需要返回 "None" 或其他默认值
    
    # 检查输入是否为字符串类型
    if not isinstance(s, str):
        # 如果不是字符串，尝试直接转换为浮点数
        try:
            num = float(s)
            return "{:.6f}".format(num)
        except (ValueError, TypeError):
            # 如果转换失败，返回原值的字符串形式或默认值
            return str(s) if s is not None else ""
    
    # 如果是字符串，去除首尾空格
    s_clean = s.strip()
    
    # 处理空字符串
    if not s_clean:
        return ""
    
    # 尝试转换为浮点数
    try:
        num = float(s_clean)
        # 格式化为6位小数
        return "{:.6f}".format(num)
    except ValueError:
        # 如果转换失败，返回原字符串（或清理后的字符串）
        return s_clean

def extract_final_answer(string: str, extract_from_boxed: bool = True,
                         extract_regex: str = r"The final answer is (.+)$"):
    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx: right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left): -1]
        except AssertionError:
            return None

    return None


def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(output_path, data_list):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            
def process_math(input_data):
    device = "cuda"
    model= AutoModelForCausalLM.from_pretrained(
        './model_math', 
        dtype=torch.float16, 
        trust_remote_code=True
    ).cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("./model_math", padding_side='left')

    outs = []
    
    lens = len(input_data)
    batch_size = 16  # 更改变量名以避免与函数冲突
    for index in range(0, lens, batch_size):
        print(f"Processing batch starting at index {index}")
    
        # 确定当前batch的实际大小（处理最后一个batch可能不足的情况）
        current_batch_size = min(batch_size, lens - index)
        
        tags = []
        messages = []
        batch_items = []  # 保存当前batch的原始数据
        texts = []
        
        for add in range(current_batch_size):
            obj = input_data[index + add]
            batch_items.append(obj)  # 保存原始对象引用        
        
            prompt = obj.get("query", "")
            tag = obj.get('tag', '填空题')
            tags.append(tag)
            # # CoT  score:64.4
            # messages = [
            #     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}. Do not use the given solution(if included)."},
            #     {"role": "user", "content": prompt}
            # ]

            # Opt Cot score:67.0
            # messages = [
            #     {
            #         "role": "system",
            #         "content": """You are an expert math problem solver. Your task is to solve the problem step by step based ONLY on the problem description provided.
            #         **Instructions:**
            #         1. **Reason step by step:** Clearly explain each step of your reasoning process. Do not skip logical steps.
            #         2. **Ignore given solutions:** If the problem includes any reference to a solution or answer (e.g., "given solution"), completely ignore it and solve independently.
            #         3. **Final answer format:** Place your final answer within \\boxed{} at the end of your response. For example: \\boxed{42}.
            #         4. **Work from first principles:** Base your solution solely on the problem statement and standard mathematical knowledge.

            #         Important: Do not assume any information not explicitly stated in the problem."""
            #     },
            #     {
            #         "role": "user", 
            #         "content": prompt  # 这里 prompt 是从图像识别步骤得到的问题描述
            #     }
            # ]

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert math problem solver. Solve this math problem using only the original problem statement. Ignore any answers or solutions that may be included. Show your reasoning step by step and put the final answer within \\boxed{}."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)        
            texts.append(text)   

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, response in enumerate(responses):
            pred_answer = extract_final_answer(response)
            try:
                match = re.search(r'\b([ABCDEF])\b', pred_answer)
                if match:
                    pred_answer = match.group(1)
                else:
                    pred_answer = str(latex2sympy(pred_answer).evalf())
            except Exception as e:
                pass
            
            # print(pred_answer)
            ans = format_string_value(pred_answer)
            ans = recheck_answer(ans, batch_items[i]["tag"])
            out = {
                "image": batch_items[i]["image"],
                "tag":  batch_items[i]["tag"],
                "step": "",
                "answer": ans
            }
            #print(out)
            outs.append(out)  
    return outs      

if __name__ == "__main__":
    start_time = time.time()
    
    input_data = load_jsonl(sys.argv[1])
    outs = process_math(input_data)
    write_jsonl(sys.argv[2], outs)
    
    print("Total Time: ", time.time() - start_time)