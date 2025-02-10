#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import random
import string
from statistics import mean
import torch
import gc
import os
import glob
import re


import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from openai_backTranslation import generate
from utils import append2file, str_to_list, convert_ABCDE, create_choices
from dpps.jointEM import joint
from dpps.SLM import SLM



def locate_result_file(file_order, search_dir='.'):
    """
    定位文件名符合 "res_{file_order}_*.txt" 模式的文件，其中:
      - file_order 对应文件名中第二部分（例如 "0.1"）
      - 文件名以 "res_" 开头，以 ".txt" 结尾，中间用下划线分隔
    参数:
      file_order: 字符串，文件顺序，如 "0.1"
      search_dir: 搜索目录，默认为当前目录
    返回:
      匹配的第一个文件完整路径，如果没有匹配则返回 None
    """
    # 构造搜索模式，例如 "res_0.1_*.txt"
    pattern = os.path.join(search_dir, f"res_{file_order}_*.txt")
    matching_files = glob.glob(pattern)
    if matching_files:
        result_file = matching_files[0]  # 如果存在多个匹配项，这里仅取第一个
        print("找到文件：", result_file)
        return result_file
    else:
        print("未找到匹配的文件。")
        return None


def process_paraphrased_questions(folder):
    # 构造匹配所有 paraphrased_questions_T{xxx}.json 文件的模式
    pattern = os.path.join(folder, 'paraphrased_questions_T*.json')
    # 使用 glob 获取所有匹配的文件路径
    file_list = glob.glob(pattern)
    
    # 遍历所有匹配的文件
    for file_path in file_list:
        # 提取文件名中的 xxx 作为 file_order
        filename = os.path.basename(file_path)
        match = re.search(r'paraphrased_questions_T(\d+(?:\.\d+)?)\.json', filename)
        if not match:
            print(f"无法从文件名 {filename} 中提取 file_order")
            continue
        
        file_order = match.group(1)
        
        # 使用 pandas 读取 json 文件，返回 DataFrame 格式
        try:
            df = pd.read_json(file_path)
        except ValueError as e:
            print(f"读取 {file_path} 出现错误: {e}")
            continue
        res_path = locate_result_file(file_order, folder)
        remove_content_after_marker(res_path)
        step1(df, res_path)

def build_answer_prompt_vqa(df: pd.DataFrame, i, t):
    """VQA 的答案 prompt：利用 OCR 提取的 words 列表"""
    prompt = ""
    for word in df.loc[i]["words"]:
        prompt += word + ", "
    prompt = (
        "Extracted OCR tokens from image:\n"
        + prompt[:-2]
        + "\nQuestion: "
        + df.loc[i][f"T_{t}"]
        + "\nAnswer the question with short term:\n"
    )
    return prompt

def remove_content_after_marker(file_path, marker='=================================================='):
    """
    删除文件中 marker 之后的所有内容，保留 marker 本身及其之前的内容。
    
    参数:
      file_path: txt 文件路径
      marker: 用于定位截断位置的标记字符串，默认为 '=================================================='
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return

    marker_index = content.find(marker)
    if marker_index != -1:
        # 保留 marker 这一段（包含 marker 本身）
        new_content = content[:marker_index + len(marker)]
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"已处理文件 {file_path}，marker 之后的内容已删除。")
        except Exception as e:
            print(f"写入文件 {file_path} 出错: {e}")
    else:
        print(f"在文件 {file_path} 中未找到 marker。")

    

def step1(paraphrased_df : pd.DataFrame, res_path):
    gt_answer = []
    for i in range(len(paraphrased_df)):
        ans = ""
        for word in paraphrased_df.loc[i]["original_answer"]:
            ans += word + ", "
        ans = ans[:-2]
        gt_answer.append(ans)
    print(gt_answer)

    rouge1_a_scores, rouge2_a_scores, rougeL_a_scores, rougeLSum_a_scores = [], [], [], []
    bleu_a_scores = []
    for temp in TEMPERATURE_LIST:
        T_predictions = []
        for i in range(len(paraphrased_df)):
            prompt = build_answer_prompt_vqa(paraphrased_df, i, temp)
            T_predictions.append(generate(prompt, temperature=0.0, stop=["\n"]))
        score = rouge_metric.compute(references=gt_answer, predictions=T_predictions)
        rouge1_a_scores.append(score["rouge1"])
        rouge2_a_scores.append(score["rouge2"])
        rougeL_a_scores.append(score["rougeL"])
        rougeLSum_a_scores.append(score["rougeLsum"])
        bleu_score = bleu_metric.compute(
            references=gt_answer, predictions=T_predictions
        )
        bleu_a_scores.append(bleu_score["bleu"])

    append2file(res_path, f"\nAnswer paraphrased S1:")
    append2file(
        res_path,
        f"rouge1 mean: {np.mean(rouge1_a_scores)}; rouge2 mean: {np.mean(rouge2_a_scores)}; rougeL mean: {np.mean(rougeL_a_scores)}; rougeLSum mean: {np.mean(rougeLSum_a_scores)}",
    )
    append2file(
        res_path,
        f"rouge1 std: {np.std(rouge1_a_scores)}; rouge2 std: {np.std(rouge2_a_scores)}; rougeL std: {np.std(rougeL_a_scores)}; rougeLSum std: {np.std(rougeLSum_a_scores)}",
    )
    append2file(res_path, f"bleu mean: {np.mean(bleu_a_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_a_scores)}")
    append2file(res_path, ">" * 50)


rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

TEMPERATURE_LIST = [1,2,3]
paths = ['results/VQA/SLM_Llama8B']

process_paraphrased_questions(paths[0])


