import os
from tqdm import tqdm
import json
import re
import torch
import sys
import time
from qwen3_vl_ocr import process_ocr
from qwen_math import process_math

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(output_path, data_list):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
def process_miniresult(result_data, rest_data):
    total_results = result_data
    
    for item in rest_data:
        if item["tag"] == "选择题":
            sample_result =  {
                "image": item["image"],
                "tag":  item["tag"],
                "step": "",
                "answer": "B"
            }
        else:
            sample_result =  {
                "image": item["image"],
                "tag":  item["tag"],
                "step": "",
                "answer": "1.000000"
            }
            
        total_results.append(sample_result)
    
    return total_results

if __name__ == "__main__":
    image_dir = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    inter_file = "generate.jsonl"
    
    process_cnt = 1000
    all_data = load_jsonl(sys.argv[2])
    
    if process_cnt > len(all_data):
        process_cnt = len(all_data)
        
    print("Total Cnt: ", len(all_data), ", Process Cnt: ", process_cnt)
    
    input_data = all_data[0:process_cnt]
    rest_data = all_data[process_cnt:]
    
    start_time = time.time()
    
    # ocr
    ocr_outs = process_ocr(image_dir, input_data, batch_size=8)
    # write_jsonl(inter_file, ocr_outs)
    ocr_time = time.time()
    print("Ocr Time: ", ocr_time - start_time)
    
    # math
    # math_data = load_jsonl(inter_file)
    math_outs = process_math(ocr_outs)
    math_time = time.time()
    print("Math Time: ", math_time - ocr_time)
    
    total_results = process_miniresult(math_outs, rest_data)
    write_jsonl(output_file, total_results)