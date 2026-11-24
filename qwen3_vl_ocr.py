from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys
import time
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_output_prompt(tag):
    """构建更精确的输出提示"""
    if tag == "选择题":
        return "The current question is a multiple-choice question. Extract ALL options exactly as shown, including option labels (A, B, C, D, etc.)."
    elif tag == "填空题":
        return "The current question is a fill-in-the-blank question. Focus on identifying the blank positions and required format."
    else:
        return "The current question is a calculation word problem. Extract the complete problem statement with all numerical values and relationships."

def load_jsonl(input_file):
    """加载jsonl文件，增加错误处理"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        raise

def write_jsonl(output_path, data_list):
    """写入jsonl文件，增加错误处理"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"成功写入 {len(data_list)} 条数据到 {output_path}")
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        raise

# ===== 删除 resize_image 函数，让Qwen3-VL自动处理 =====

def create_optimized_prompt():
    """Create optimized prompt"""
    return """
    You are a math problem extraction expert. Your only task is to extract the complete math problem from the image.

    Critical Instructions:
    1. Extract ALL text: Transcribe every word, number, and symbol including:
       - Full question statement
       - All equations and formulas  
       - Multiple-choice options with exact labels
       - Diagram labels and captions
       - Unit symbols and notations

    2. Preserve exact formatting: Maintain original mathematical notation, fractions, exponents, and symbols

    3. Ignore: Handwritten solutions, stray marks, or working steps

    4. Output format:
       Problem: [Complete verbatim transcription]
       Options: [Only if multiple-choice, list all options with labels]

    Important: Do not solve the problem! Include the complete problem text. If unsure about any element, include it.
    """
    
    # """创建优化的提示词"""
    # return """
    # You are a specialized math problem extraction expert. Your ONLY task is to extract the complete mathematical problem from the image.

    # CRITICAL INSTRUCTIONS:
    # 1. EXTRACT ALL TEXT: Transcribe EVERY word, number, and symbol from the math problem including:
    #    - Full question statement
    #    - All equations and formulas
    #    - Multiple-choice options (if present) with their exact labels (A, B, C, D, etc.)
    #    - Diagrams' textual labels and captions
    #    - Unit symbols and special notations

    # 2. PRESERVE EXACT FORMATTING:
    #    - Maintain original mathematical notation
    #    - Keep fractions, exponents, and special symbols intact
    #    - Preserve option ordering and labeling
    #    - Include all numerical values exactly as shown

    # 3. COMPLETELY IGNORE:
    #    - Handwritten solutions or answers
    #    - Stray marks or unrelated text
    #    - Solution steps or working
    #    - Any text that is not part of the problem statement

    # 4. OUTPUT FORMAT:
    #    Problem: [Complete verbatim transcription of the entire math problem with all components]
    #    Options: [Only if multiple-choice, list all options with their labels exactly as shown]

    # IMPORTANT: Do not try to solve the problem!Your response must contain the COMPLETE problem. If unsure about any element, include it.
    # """
    
    
    

def clean_model_response(response, tag):
    """改进的响应清理函数"""
    # 移除重复的提示词内容
    patterns_to_remove = [
        r'You are a specialized math problem extraction expert.*?OUTPUT FORMAT:',
        r'CRITICAL INSTRUCTIONS:.*?IMPORTANT:',
        r'Your ONLY task is to extract.*?IMPORTANT:',
    ]
    
    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # 根据题目类型进行特定清理
    if tag == "选择题":
        # 确保选项被完整保留
        if "Options:" not in response and re.search(r'[A-D][\.\)]', response):
            # 自动添加Options标签如果检测到选项但缺少标签
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if re.search(r'[A-D][\.\)]', line):
                    lines.insert(i, "Options:")
                    response = '\n'.join(lines)
                    break
    else:
        # 对于非选择题，移除选项相关内容
        response = re.sub(r'Options:.*', '', response, flags=re.DOTALL)
    
    # 清理多余的空白和换行
    response = re.sub(r'\n\s*\n', '\n', response)  # 多个空行合并为一个
    response = response.strip()
    
    return response

def validate_extraction(response, tag):
    """验证提取的完整性"""
    if not response or len(response.strip()) < 10:  # 响应太短
        return False, "响应内容过短"
    
    # 检查是否包含数学相关的内容
    math_indicators = [r'\d+', r'[+\-*/=]', r'[a-zA-Z]\s*=', r'选择', r'填空', r'计算', r'问题']
    has_math_content = any(re.search(pattern, response) for pattern in math_indicators)
    
    if not has_math_content:
        return False, "未检测到数学相关内容"
    
    # 对于选择题，检查是否有选项
    if tag == "选择题" and not re.search(r'[A-D][\.\)]', response):
        return False, "选择题但未检测到选项"
    
    return True, "验证通过"

def process_batch(model, processor, batch_data, image_dir, question):
    """改进的批量处理函数 - Qwen3-VL自动处理图片"""
    batch_size = len(batch_data)
    
    # 准备批量输入
    messages_batch = []
    tags = []
    image_paths = []
    
    for obj in batch_data:
        image_path = os.path.join(image_dir, obj['image'])
        tag = obj.get('tag', '填空题')
        
        # ===== 直接传图片路径，不做resize =====
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},  # 直接传路径字符串
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        messages_batch.append(messages)
        tags.append(tag)
        image_paths.append(obj['image'])
    
    try:
        max_tokens = 1024
        
        # ===== Qwen3-VL的批量预处理 =====
        texts = []
        all_image_inputs = []
        with torch.no_grad():
            for messages in messages_batch:
                # 应用聊天模板
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                texts.append(text)
                
                # 处理图像 - Qwen3-VL会自动读取和resize
                image_inputs, _ = process_vision_info(messages)
                if image_inputs:
                    all_image_inputs.extend(image_inputs)
            
            # 使用processor处理
            inputs = processor(
                text=texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
        # ===== 批量生成 =====

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            
            # ===== 去除输入部分 =====
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            
            # ===== 使用processor解码 =====
            responses = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        # 处理批量输出（逻辑不变）
        results = []
        for i in range(batch_size):
            response = responses[i]
            
            tag = tags[i]
            # 清理响应
            cleaned_response = clean_model_response(response, tag)
            
            # 验证提取质量
            is_valid, validation_msg = validate_extraction(cleaned_response, tag)
            
            output_entry = {
                "image": image_paths[i],
                "query": build_output_prompt(tag) + cleaned_response,
                "tag": tag,
                "validation": {
                    "is_valid": is_valid,
                    "message": validation_msg,
                    "original_length": len(response),
                    "cleaned_length": len(cleaned_response)
                }
            }
            results.append(output_entry)
            
            if not is_valid:
                logger.warning(f"提取验证失败: {image_paths[i]} - {validation_msg}")
        
        return results
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        # 返回失败结果
        return [{
            "image": image_paths[i],
            "query": f"处理失败: {str(e)}",
            "tag": tags[i],
            "validation": {
                "is_valid": False,
                "message": f"处理异常: {str(e)}",
                "original_length": 0,
                "cleaned_length": 0
            }
        } for i in range(batch_size)]

def process_ocr(image_dir, input_data, batch_size=4):
    """主处理函数"""
    
    # 使用优化的提示词
    question = create_optimized_prompt()
    
    # ===== 加载Qwen3-VL模型 =====
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            './model', 
            torch_dtype=torch.float16,  # 使用float16
            trust_remote_code=True,
            device_map="auto"
        )
        model.eval()
        
        processor = AutoProcessor.from_pretrained('./model', padding_side='left')
        
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    
    output_lines = []
    success_count = 0
    total_count = len(input_data)
    
    logger.info(f"开始处理 {total_count} 张图片，批次大小: {batch_size}")
    
    for i in tqdm(range(0, len(input_data), batch_size), desc="处理批次"):
        batch_data = input_data[i:i+batch_size]
        batch_results = process_batch(model, processor, batch_data, image_dir, question)
        output_lines.extend(batch_results)
        
        # 统计成功数量
        success_count += sum(1 for result in batch_results if result["validation"]["is_valid"])
    
    success_rate = (success_count / total_count) * 100
    logger.info(f"处理完成: 成功 {success_count}/{total_count} ({success_rate:.1f}%)")
    
    return output_lines

def analyze_results(output_lines):
    """分析处理结果"""
    valid_count = sum(1 for item in output_lines if item["validation"]["is_valid"])
    total_count = len(output_lines)
    
    logger.info(f"结果分析:")
    logger.info(f"  总处理数: {total_count}")
    logger.info(f"  有效提取: {valid_count} ({valid_count/total_count*100:.1f}%)")
    logger.info(f"  失败数量: {total_count - valid_count}")
    
    # 按题目类型统计
    tag_stats = {}
    for item in output_lines:
        tag = item["tag"]
        if tag not in tag_stats:
            tag_stats[tag] = {"total": 0, "valid": 0}
        tag_stats[tag]["total"] += 1
        if item["validation"]["is_valid"]:
            tag_stats[tag]["valid"] += 1
    
    for tag, stats in tag_stats.items():
        valid_rate = stats["valid"] / stats["total"] * 100
        logger.info(f"  {tag}: {stats['valid']}/{stats['total']} ({valid_rate:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python script.py <图片目录> <输入jsonl文件> <输出jsonl文件>")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        input_data = load_jsonl(sys.argv[2])
        output_lines = process_ocr(sys.argv[1], input_data, batch_size=4)
        write_jsonl(sys.argv[3], output_lines)
        
        # 分析结果
        analyze_results(output_lines)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)
    
    total_time = time.time() - start_time
    logger.info(f"总执行时间: {total_time:.2f} 秒")
    logger.info(f"平均每张图片: {total_time/len(input_data):.2f} 秒")