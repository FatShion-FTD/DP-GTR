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

import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from openai_backTranslation import generate
from utils import append2file, str_to_list, convert_ABCDE, create_choices
from dpps.jointEM import joint
from dpps.SLM import SLM

# 下载 NLTK 资源（首次运行时可能需要）
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from nltk.util import ngrams

# 全局参数
TEST_SIZE = 200
# 每个实验中采用的不同温度列表（可根据需要修改）
TEMPERATURE_LIST = [1,2,3,4,5,6,7,8,9,10] # [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# 对应实验中使用的 logits bias 字典
LOGITS_BIAS_DICT = {"32": 100, "33": 100, "34": 100, "35": 100, "36": 100}
# 其他参数（例如 REMAIN 用于提示生成时避免的 token 数量）
REMAIN = 10

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SENSITIVITY = 9.7
DATASET_TYPE = None # "csQA", "medQA", "VQA"
model = SLM(MODEL_NAME)


rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
perplexity_metric = evaluate.load("perplexity", module_type="metric")
acc_metric = evaluate.load("accuracy")





def build_answer_prompt_csqa(df: pd.DataFrame, i, t):
    """csQA 的答案 prompt：遍历 choices 字典中的 label 与 text"""
    prompt = ""
    for idx in range(len(df.loc[i]['choices']['label'])):
        prompt += df.loc[i]['choices']['label'][idx] + ". " + df.loc[i]['choices']['text'][idx] + "\n"
    prompt = df.loc[i][f"T_{t}"] + "\nAnswer the question with the following options: \n" + prompt + "Answer Index: "
    return prompt

def build_answer_prompt_medqa(df: pd.DataFrame, i, t):
    """medQA 的答案 prompt：此处 choices 存在于 options 字段（作为一个字典）"""
    prompt = ""
    for k, v in df.loc[i]['choices'].items():
        prompt += k + ". " + v + "\n"
    prompt = df.loc[i][f"T_{t}"] + "\nAnswer the question with the following options: \n" + prompt + "Answer Index: "
    return prompt

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

def question_paraphrase(question):
    prompt = f"Question : {question}\nParaphrase of the question :"
    output = model.generate(prompt, prompt)['output_text']
    return model.clean_text(output, prompt)

# ------------------ csQA 实验 ------------------
def run_csqa(paraphrase_T: float):
    path_prefix = "results/csQA/SLM_Llama8B/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    EXP_NAME = f"\n{REMAIN} Tokens Avoid Generation lowest ppl reference ICL Experiment:"  # 可根据需要调整说明文字
    
    epsilon = 2 * SENSITIVITY / paraphrase_T
    model.clip_model(epsilon=epsilon, clip_type="all_clip")
    
    
    # 加载 csQA 数据集（如果 N==5 则采用 commonsense_qa 数据集）
    N = 5
    data = load_dataset("tau/commonsense_qa")
    test_df = data['validation'].to_pandas().head(TEST_SIZE)

    # 构造用于 paraphrase 的 DataFrame
    paraphrased_df = pd.DataFrame()
    paraphrased_df['original_question'] = test_df['question']
    paraphrased_df['original_answer'] = test_df['answerKey']
    paraphrased_df['choices'] = test_df['choices']


    # 对每个温度生成 paraphrased 文本
    for temp in TEMPERATURE_LIST:
        paraphrased_df[f"T_{temp}"] = test_df["question"].apply(
            lambda x: question_paraphrase(x)
        )

    # 删除所有 paraphrased 都为空的行
    for i in range(len(paraphrased_df)):
        tot_len = sum(len(paraphrased_df.loc[i][f"T_{temp}"]) for temp in TEMPERATURE_LIST)
        if tot_len == 0:
            paraphrased_df.drop(i, inplace=True)
            append2file(os.path.join(path_prefix, f"res_{paraphrase_T}.txt"), f"row {i}: empty!!")
    paraphrased_df.reset_index(drop=True, inplace=True)

    references = paraphrased_df["original_question"].tolist()
    rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
    bleu_scores = []
    for temp in TEMPERATURE_LIST:
        predictions = paraphrased_df[f"T_{temp}"].tolist()
        score = rouge_metric.compute(references=references, predictions=predictions)
        rouge1_scores.append(score["rouge1"])
        rouge2_scores.append(score["rouge2"])
        rougeL_scores.append(score["rougeL"])
        rougeLSum_scores.append(score["rougeLsum"])
        bleu_score = bleu_metric.compute(references=references, predictions=predictions)
        bleu_scores.append(bleu_score["bleu"])

    res_path = os.path.join(path_prefix, f"res_{paraphrase_T}_{epsilon}.txt")
    append2file(res_path, EXP_NAME)
    append2file(res_path, f"Paraphrase T: {paraphrase_T}")
    append2file(res_path, "Question paraphrased S1:")
    append2file(res_path,
                f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
                f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}")
    append2file(res_path,
                f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
                f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

    df_file_path = os.path.join(path_prefix, f"paraphrased_questions_T{paraphrase_T}.json")
    paraphrased_df.to_json(df_file_path, orient="records")
    paraphrased_df = pd.read_json(df_file_path, orient="records")

    # 生成答案
    gt_answer = convert_ABCDE(paraphrased_df["original_answer"].tolist())
    acc_scores = []
    for temp in TEMPERATURE_LIST:
        T_predictions = []
        for i in range(len(paraphrased_df)):
            prompt = build_answer_prompt_csqa(paraphrased_df, i, temp)
            T_predictions.append(generate(prompt, temperature=0.0, logits_dict=LOGITS_BIAS_DICT, max_tokens=1))
        T_predictions = convert_ABCDE(T_predictions)
        acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
        acc_scores.append(acc_score['accuracy'])
    append2file(res_path, "Answer paraphrased S1:")
    append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
    append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
    append2file(res_path, ">" * 50)

    step2_csQA_medQA(paraphrased_df, res_path, path_prefix, paraphrase_T)


# ------------------ medQA 实验 ------------------
def run_medqa(paraphrase_T: float):
    path_prefix = "results/medQA/SLM_Llama8B/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    EXP_NAME = f"\n{REMAIN} Tokens Avoid Generation lowest ppl reference ICL Experiment:"
    
    epsilon = 2 * SENSITIVITY / paraphrase_T
    model.clip_model(epsilon=epsilon, clip_type="all_clip")
    
    
    # 使用 medQA 数据（假设文件 dataset/medQA_{N}.json 存在，此处 N=4）
    N = 4
    data = pd.read_json(f"dataset/medQA_{N}.json", orient="records")
    test_df = data.head(TEST_SIZE)

    paraphrased_df = pd.DataFrame()
    paraphrased_df['original_question'] = test_df['question']
    paraphrased_df['original_answer'] = test_df['answer_idx']
    paraphrased_df['choices'] = test_df['options']

    for temp in TEMPERATURE_LIST:
        paraphrased_df[f"T_{temp}"] = test_df["question"].apply(
            lambda x: question_paraphrase(x)
        )

    for i in range(len(paraphrased_df)):
        tot_len = sum(len(paraphrased_df.loc[i][f"T_{temp}"]) for temp in TEMPERATURE_LIST)
        if tot_len == 0:
            paraphrased_df.drop(i, inplace=True)
            append2file(os.path.join(path_prefix, f"res_{paraphrase_T}.txt"), f"row {i}: empty!!")
    paraphrased_df.reset_index(drop=True, inplace=True)

    references = paraphrased_df["original_question"].tolist()
    rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
    bleu_scores = []
    for temp in TEMPERATURE_LIST:
        predictions = paraphrased_df[f"T_{temp}"].tolist()
        score = rouge_metric.compute(references=references, predictions=predictions)
        rouge1_scores.append(score["rouge1"])
        rouge2_scores.append(score["rouge2"])
        rougeL_scores.append(score["rougeL"])
        rougeLSum_scores.append(score["rougeLsum"])
        bleu_score = bleu_metric.compute(references=references, predictions=predictions)
        bleu_scores.append(bleu_score["bleu"])
    res_path = os.path.join(path_prefix, f"res_{paraphrase_T}_{epsilon}.txt")
    append2file(res_path, EXP_NAME)
    append2file(res_path, f"Paraphrase T: {paraphrase_T}")
    append2file(res_path,
                f"Question paraphrased S1: rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
                f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}")
    append2file(res_path,
                f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
                f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

    df_file_path = os.path.join(path_prefix, f"paraphrased_questions_T{paraphrase_T}.json")
    paraphrased_df.to_json(df_file_path, orient="records")
    paraphrased_df = pd.read_json(df_file_path, orient="records")

    gt_answer = []
    for i in range(len(paraphrased_df)):
        ans = ""
        for word in paraphrased_df.loc[i]["original_answer"]:
            ans += word + ", "
        ans = ans[:-2]
        gt_answer.append(ans)
    gt_answer = convert_ABCDE(gt_answer)

    acc_scores = []
    for temp in TEMPERATURE_LIST:
        T_predictions = []
        for i in range(len(paraphrased_df)):
            prompt = build_answer_prompt_medqa(paraphrased_df, i, temp)
            T_predictions.append(generate(prompt, temperature=0.0, logits_dict=LOGITS_BIAS_DICT, max_tokens=1))
        T_predictions = convert_ABCDE(T_predictions)
        acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
        acc_scores.append(acc_score['accuracy'])
    append2file(res_path, "Answer paraphrased S1:")
    append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
    append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
    append2file(res_path, ">" * 50)

    step2_csQA_medQA(paraphrased_df, res_path, path_prefix, paraphrase_T)
    
    

# ------------------ VQA 实验 ------------------
def run_vqa(paraphrase_T: float):
    path_prefix = "results/VQA/SLM_Llama8B/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    EXP_NAME = f"\n{REMAIN} Tokens Avoid Generation lowest ppl reference ICL Experiment:"
    
    epsilon = 2 * SENSITIVITY / paraphrase_T
    model.clip_model(epsilon=epsilon, clip_type="all_clip")

    # 加载 VQA 数据集（docvqa 1200 例子）
    data = load_dataset("nielsr/docvqa_1200_examples_donut")
    test_df = data["test"].to_pandas().head(TEST_SIZE)
    paraphrased_df = pd.DataFrame()
    paraphrased_df['original_question'] = test_df['query'].apply(lambda x: x['en'])
    paraphrased_df["original_answer"] = test_df["answers"]
    paraphrased_df['words'] = test_df['words']

    for temp in TEMPERATURE_LIST:
        paraphrased_df[f"T_{temp}"] = paraphrased_df['original_question'].apply(
            lambda x: question_paraphrase(x)
        )

    for i in range(len(paraphrased_df)):
        tot_len = sum(len(paraphrased_df.loc[i][f"T_{temp}"]) for temp in TEMPERATURE_LIST)
        if tot_len == 0:
            paraphrased_df.drop(i, inplace=True)
            append2file(os.path.join(path_prefix, f"res_{paraphrase_T}.txt"), f"row {i}: empty!!")
    paraphrased_df.reset_index(drop=True, inplace=True)

    references = paraphrased_df["original_question"].tolist()
    rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
    bleu_scores = []
    for temp in TEMPERATURE_LIST:
        predictions = paraphrased_df[f"T_{temp}"].tolist()
        score = rouge_metric.compute(references=references, predictions=predictions)
        rouge1_scores.append(score["rouge1"])
        rouge2_scores.append(score["rouge2"])
        rougeL_scores.append(score["rougeL"])
        rougeLSum_scores.append(score["rougeLsum"])
        bleu_score = bleu_metric.compute(references=references, predictions=predictions)
        bleu_scores.append(bleu_score["bleu"])
    res_path = os.path.join(path_prefix, f"res_{paraphrase_T}_{epsilon}.txt")
    append2file(res_path, EXP_NAME)
    append2file(res_path, f"Paraphrase T: {paraphrase_T}")
    append2file(res_path,
                f"Question paraphrased S1: rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
                f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}")
    append2file(res_path,
                f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
                f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

    df_file_path = os.path.join(path_prefix, f"paraphrased_questions_T{paraphrase_T}.json")
    paraphrased_df.to_json(df_file_path, orient="records")
    paraphrased_df = pd.read_json(df_file_path, orient="records")

    gt_answer = []
    for i in range(len(paraphrased_df)):
        ans = ""
        for word in paraphrased_df.loc[i]["original_answer"]:
            ans += word + ", "
        ans = ans[:-2]
        gt_answer.append(ans)
    print(gt_answer)
    gt_answer = convert_ABCDE(gt_answer)
    acc_scores = []
    for temp in TEMPERATURE_LIST:
        T_predictions = []
        for i in range(len(paraphrased_df)):
            prompt = build_answer_prompt_vqa(paraphrased_df, i, temp)
            T_predictions.append(generate(prompt, temperature=0.0, stop=["\n"]))
        T_predictions = convert_ABCDE(T_predictions)
        acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
        acc_scores.append(acc_score['accuracy'])
    append2file(res_path, "Answer paraphrased S1:")
    append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
    append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
    append2file(res_path, ">" * 50)

    step2_vqa(paraphrased_df, res_path, path_prefix, paraphrase_T)



def step2_vqa(paraphrased_df:pd.DataFrame, res_path, path_prefix, paraphrase_T):
    questions_gt_list = paraphrased_df["original_question"].tolist()
    answers_gt_list = paraphrased_df["original_answer"].tolist()
    paraphrased_df = paraphrased_df.fillna(" ")
    
    import nltk
    nltk.download("stopwords")
    nltk.download("punkt")
    from nltk.corpus import stopwords
    from nltk.util import ngrams

    stopword_set = set(stopwords.words("english"))
    import string
    import random

    repeats = 3
    (
        bleu_q_scores,
        rouge1_q_scores,
        rouge2_q_scores,
        rougeL_q_scores,
        rougeLSum_q_scores
    ) = ([], [], [], [], [])
    
    bleu_q_jointEM_scores = []
    rouge1_q_jointEM_scores = []
    rouge2_q_jointEM_scores = []
    rougeL_q_jointEM_scores = []
    rougeLSum_q_jointEM_scores = []
    
    
    rouge1_ndqa_scores = []
    rouge2_ndqa_scores = []
    rougeL_ndqa_scores = []
    rougeLSum_ndqa_scores = []
    bleu_ndqa_scores = []
    rouge1_jema_scores = []
    rouge2_jema_scores = []
    rougeL_jema_scores = []
    rougeLSum_jema_scores = []
    bleu_jema_scores = []
    
    # add anaysis columns to paraphrased_df
    paraphrased_df['ndp_tokens'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_tokens'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['reference_question'] = ""
    paraphrased_df['ndp_question'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_question'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['ndp_answer'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_answer'] = [[] for _ in range(len(paraphrased_df))]
    
    
    for r in range(repeats):
        questions_list = []
        answers_list = []
        jem_questions_list = []
        jem_answers_list = []
        
        for i in range(len(paraphrased_df)):
            # Count token freq
            all_tokens = {}  # key: token, value: count
            for t in TEMPERATURE_LIST:
                sentence = paraphrased_df.loc[i][f"T_{t}"]
                tokens = nltk.word_tokenize(sentence)
                onegrams = set(ngrams(tokens, 1))
                for token in onegrams:
                    # only add one gram for one sentence
                    if token in all_tokens:
                        all_tokens[token] += 1
                    else:
                        all_tokens[token] = 1
            print(f"All Tokens:  {all_tokens}")

            # ================ Add Noise Here ================
            all_tokens_sorted = sorted(
                all_tokens.items(), key=lambda x: x[1], reverse=True
            )
            print(f"All Sorted Tokens:  {all_tokens_sorted}")
            # ignore those non-words tokens
            filtered_tokens = {}
            for token, count in all_tokens_sorted:
                if (
                    not all(word in string.punctuation for word in token)
                    and token[0] not in stopword_set
                ):
                    filtered_tokens[token] = count
            filtered_tokens_sorted = sorted(
                filtered_tokens.items(), key=lambda x: x[1], reverse=True
            )
            print(f"Filtered Sorted Tokens:  {filtered_tokens_sorted}")

            # TOP 10 tokens
            filtered_tokens_sorted_ndp = filtered_tokens_sorted[:min(REMAIN, len(filtered_tokens_sorted))]
            filtered_tokens_ndp = [k[0][0] for k in filtered_tokens_sorted_ndp]
            print(filtered_tokens_ndp)
            
            # JointEM
            item_counts = np.array([count for token, count in filtered_tokens_sorted])
            joint_out = joint(item_counts, k=min(REMAIN, len(item_counts)), epsilon=2, neighbor_type=1)
            filtered_tokens_jem = np.array(filtered_tokens_sorted, dtype=object)[joint_out]
            filtered_tokens_jem = [token_tuple[0][0] for token_tuple in filtered_tokens_jem]
            print(filtered_tokens_jem)
            
            # lowest ppl reference question
            paraphrase_sentences = []
            for t in TEMPERATURE_LIST:
                if len(paraphrased_df.loc[i][f"T_{t}"]) > 0:
                    paraphrase_sentences.append(paraphrased_df.loc[i][f"T_{t}"])
                else:
                    paraphrase_sentences.append(" ")
                    
            perplexity_res = perplexity_metric.compute(predictions=paraphrase_sentences, model_id="gpt2")
            tmp_df = pd.DataFrame({"Predictions": paraphrase_sentences, "Perplexity": perplexity_res['perplexities']})
            lowest_perplexity_idx = tmp_df["Perplexity"].idxmin()
            reference_question = tmp_df.loc[lowest_perplexity_idx]["Predictions"]
            # ================ End Here ================

            random.shuffle(filtered_tokens_jem)  
            random.shuffle(filtered_tokens_ndp)  
            
            # For Non-DP
            # Build tokens prompt
            suggest_tokens = ""
            for token in filtered_tokens_ndp:
                suggest_tokens += token + ", "
            suggest_tokens = suggest_tokens[:-2]

            # Build Prompt and generate questions
            icl_prompt = (
                "Refer the following question to generate a new question:\n"
                + reference_question
                + "\nAvoid using following tokens:\n"
                # + "\nDo not using following tokens:\n"
                + suggest_tokens
                + "\nGenerated question:\n"
            )
            question = model.generate(icl_prompt, reference_question)['output_text']
            question = model.clean_text(question, icl_prompt)

            ## cloud enable section ##
            # Generate answers
            prompt = ""
            for word in paraphrased_df.loc[i]["words"]:
                prompt += word + ", "
            prompt = (
                "Extracted OCR tokens from image:\n"
                + prompt[:-2]
                + "\nQuestion: "
                + question
                + "\nAnswer the question with short term:\n"
            )
            answers = generate(prompt, temperature=0.0, stop=["\n"])

            questions_list.append(question)
            answers_list.append(answers)
            
            if len(paraphrased_df.loc[i]['reference_question']) == 0:
                paraphrased_df.loc[i, 'reference_question'] = reference_question
            paraphrased_df.loc[i, 'ndp_tokens'].append(filtered_tokens_ndp)
            paraphrased_df.loc[i, 'ndp_question'].append(question)
            paraphrased_df.loc[i, 'ndp_answer'].append(answers)
            
            # For JointEM
            # Build tokens prompt
            suggest_tokens = ""
            for token in filtered_tokens_jem:
                suggest_tokens += token + ", "
            suggest_tokens = suggest_tokens[:-2]
            
            # Build Prompt and generate questions
            icl_prompt = (
                "Refer the following question to generate a new question:\n"
                + reference_question
                + "\nAvoid using following tokens:\n"
                # + "\nDo not using following tokens:\n"
                + suggest_tokens
                + "\nGenerated question:\n"
            )
            question = model.generate(icl_prompt, reference_question)['output_text']
            question = model.clean_text(question, icl_prompt)
            
            ## cloud enable section ##
            # Generate answers
            prompt = ""
            for word in paraphrased_df.loc[i]["words"]:
                prompt += word + ", "
            prompt = (
                "Extracted OCR tokens from image:\n"
                + prompt[:-2]
                + "\nQuestion: "
                + question
                + "\nAnswer the question with short term:\n"
            )
            answers = generate(prompt, temperature=0.0, stop=["\n"])
            
            jem_questions_list.append(question)
            jem_answers_list.append(answers)
            
            paraphrased_df.loc[i, 'jem_tokens'].append(filtered_tokens_jem)
            paraphrased_df.loc[i, 'jem_question'].append(question)
            paraphrased_df.loc[i, 'jem_answer'].append(answers)
            
        try:
            bleu_q_score = bleu_metric.compute(references=questions_gt_list, predictions=questions_list)
        except:
            bleu_q_score = {'bleu': 0}
        try:
            rouge_q_score = rouge_metric.compute(references=questions_gt_list, predictions=questions_list)
        except:
            rouge_q_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        try:
            rouge_a_ndp_score = rouge_metric.compute(references=answers_gt_list, predictions=answers_list)
        except:
            rouge_a_ndp_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        try:
            bleu_a_score = bleu_metric.compute(references=answers_gt_list, predictions=answers_list)
        except:
            bleu_a_score = {'bleu': 0}    
            
        bleu_q_scores.append(bleu_q_score["bleu"])
        rouge1_q_scores.append(rouge_q_score["rouge1"])
        rouge2_q_scores.append(rouge_q_score["rouge2"])
        rougeL_q_scores.append(rouge_q_score["rougeL"])
        rougeLSum_q_scores.append(rouge_q_score["rougeLsum"])
        
        rouge1_ndqa_scores.append(rouge_a_ndp_score["rouge1"])
        rouge2_ndqa_scores.append(rouge_a_ndp_score["rouge2"])
        rougeL_ndqa_scores.append(rouge_a_ndp_score["rougeL"])
        rougeLSum_ndqa_scores.append(rouge_a_ndp_score["rougeLsum"])
        bleu_ndqa_scores.append(bleu_a_score["bleu"])
        
        try:
            bleu_q_jem_score = bleu_metric.compute(references=questions_gt_list, predictions=jem_questions_list)
        except:
            bleu_q_jem_score = {'bleu': 0}
        try:
            rouge_q_jem_score = rouge_metric.compute(references=questions_gt_list, predictions=jem_questions_list)
        except:
            rouge_q_jem_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        try:
            rouge_a_jem_score = rouge_metric.compute(references=answers_gt_list, predictions=jem_answers_list)
        except:
            rouge_a_jem_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        try:
            bleu_a_score = bleu_metric.compute(references=answers_gt_list, predictions=jem_answers_list)
        except:
            bleu_a_score = {'bleu': 0}
        
        bleu_q_jointEM_scores.append(bleu_q_jem_score["bleu"])
        rouge1_q_jointEM_scores.append(rouge_q_jem_score["rouge1"])
        rouge2_q_jointEM_scores.append(rouge_q_jem_score["rouge2"])
        rougeL_q_jointEM_scores.append(rouge_q_jem_score["rougeL"])
        rougeLSum_q_jointEM_scores.append(rouge_q_jem_score["rougeLsum"])
        
        rouge1_jema_scores.append(rouge_a_jem_score["rouge1"])
        rouge2_jema_scores.append(rouge_a_jem_score["rouge2"])
        rougeL_jema_scores.append(rouge_a_jem_score["rougeL"])
        rougeLSum_jema_scores.append(rouge_a_jem_score["rougeLsum"])
        bleu_jema_scores.append(bleu_a_score["bleu"])

    append2file(res_path, f"Non-DP Question generated:")
    append2file(res_path, f"rouge1 mean: {np.mean(rouge1_q_scores)}; rouge2 mean: {np.mean(rouge2_q_scores)}; rougeL mean: {np.mean(rougeL_q_scores)}; rougeLSum mean: {np.mean(rougeLSum_q_scores)}")
    append2file(res_path, f"rouge1 std: {np.std(rouge1_q_scores)}; rouge2 std: {np.std(rouge2_q_scores)}; rougeL std: {np.std(rougeL_q_scores)}; rougeLSum std: {np.std(rougeLSum_q_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_q_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_q_scores)}")
    append2file(res_path, f"Answer rouge1 mean: {np.mean(rouge1_ndqa_scores)}; rouge2 mean: {np.mean(rouge2_ndqa_scores)}; rougeL mean: {np.mean(rougeL_ndqa_scores)}; rougeLSum mean: {np.mean(rougeLSum_ndqa_scores)}")
    append2file(res_path, f"Answer rouge1 std: {np.std(rouge1_ndqa_scores)}; rouge2 std: {np.std(rouge2_ndqa_scores)}; rougeL std: {np.std(rougeL_ndqa_scores)}; rougeLSum std: {np.std(rougeLSum_ndqa_scores)}")
    append2file(res_path, f"Answer bleu mean: {np.mean(bleu_ndqa_scores)}")
    append2file(res_path, f"Answer bleu std: {np.std(bleu_ndqa_scores)}")
    append2file(res_path, "<" * 50)
    
    append2file(res_path, f"JointEM Question generated:")
    append2file(res_path, f"rouge1 mean: {np.mean(rouge1_q_jointEM_scores)}; rouge2 mean: {np.mean(rouge2_q_jointEM_scores)}; rougeL mean: {np.mean(rougeL_q_jointEM_scores)}; rougeLSum mean: {np.mean(rougeLSum_q_jointEM_scores)}")
    append2file(res_path, f"rouge1 std: {np.std(rouge1_q_jointEM_scores)}; rouge2 std: {np.std(rouge2_q_jointEM_scores)}; rougeL std: {np.std(rougeL_q_jointEM_scores)}; rougeLSum std: {np.std(rougeLSum_q_jointEM_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_q_jointEM_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_q_jointEM_scores)}")
    append2file(res_path, f"Answer rouge1 mean: {np.mean(rouge1_jema_scores)}; rouge2 mean: {np.mean(rouge2_jema_scores)}; rougeL mean: {np.mean(rougeL_jema_scores)}; rougeLSum mean: {np.mean(rougeLSum_jema_scores)}")
    append2file(res_path, f"Answer rouge1 std: {np.std(rouge1_jema_scores)}; rouge2 std: {np.std(rouge2_jema_scores)}; rougeL std: {np.std(rougeL_jema_scores)}; rougeLSum std: {np.std(rougeLSum_jema_scores)}")
    append2file(res_path, f"Answer bleu mean: {np.mean(bleu_jema_scores)}")
    append2file(res_path, f"Answer bleu std: {np.std(bleu_jema_scores)}")
    append2file(res_path, "=" * 50)
    
    paraphrased_df.to_json(f"{path_prefix}fin_T{paraphrase_T}.json", orient="records")


def step2_csQA_medQA(paraphrased_df:pd.DataFrame, res_path, path_prefix, paraphrase_T):
    # Step2: KSA Generate Questions 
    # paraphrased_df.head()
    questions_gt_list = paraphrased_df["original_question"].tolist()
    answers_gt_list = paraphrased_df["original_answer"].tolist()
    answers_gt_list = convert_ABCDE(answers_gt_list)
    
    import nltk
    nltk.download("stopwords")
    nltk.download("punkt")
    from nltk.corpus import stopwords
    from nltk.util import ngrams

    stopword_set = set(stopwords.words("english"))
    import string
    import random

    repeats = 3

    (
        bleu_q_scores,
        rouge1_q_scores,
        rouge2_q_scores,
        rougeL_q_scores,
        rougeLSum_q_scores
    ) = ([], [], [], [], [])
    
    bleu_q_jointEM_scores = []
    rouge1_q_jointEM_scores = []
    rouge2_q_jointEM_scores = []
    rougeL_q_jointEM_scores = []
    rougeLSum_q_jointEM_scores = []
    
    acc_scores = []
    acc_jointEM_scores = []
    
    # add anaysis columns to paraphrased_df
    paraphrased_df['ndp_tokens'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_tokens'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['reference_question'] = ""
    paraphrased_df['ndp_question'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_question'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['ndp_answer'] = [[] for _ in range(len(paraphrased_df))]
    paraphrased_df['jem_answer'] = [[] for _ in range(len(paraphrased_df))]
    
    
    for _ in range(repeats):
        questions_list = []
        answers_list = []
        jem_questions_list = []
        jem_answers_list = []
        
        for i in range(len(paraphrased_df)):
            # Count token freq
            all_tokens = {}  # key: token, value: count
            for t in TEMPERATURE_LIST:
                sentence = paraphrased_df.loc[i][f"T_{t}"]
                tokens = nltk.word_tokenize(sentence)
                onegrams = set(ngrams(tokens, 1))
                for token in onegrams:
                    # only add one gram for one sentence
                    if token in all_tokens:
                        all_tokens[token] += 1
                    else:
                        all_tokens[token] = 1
            print(f"All Tokens:  {all_tokens}")

            # ================ Add Noise Here ================
            all_tokens_sorted = sorted(
                all_tokens.items(), key=lambda x: x[1], reverse=True
            )
            print(f"All Sorted Tokens:  {all_tokens_sorted}")
            # ignore those non-words tokens
            filtered_tokens = {}
            for token, count in all_tokens_sorted:
                if (
                    not all(word in string.punctuation for word in token)
                    and token[0] not in stopword_set
                ):
                    filtered_tokens[token] = count
            filtered_tokens_sorted = sorted(
                filtered_tokens.items(), key=lambda x: x[1], reverse=True
            )
            print(f"Filtered Sorted Tokens:  {filtered_tokens_sorted}")

            # TOP 10 tokens
            filtered_tokens_sorted_ndp = filtered_tokens_sorted[:min(REMAIN, len(filtered_tokens_sorted))]
            filtered_tokens_ndp = [k[0][0] for k in filtered_tokens_sorted_ndp]
            print(filtered_tokens_ndp)
            
            # JointEM
            item_counts = np.array([count for token, count in filtered_tokens_sorted])
            joint_out = joint(item_counts, k=min(REMAIN, len(item_counts)), epsilon=2, neighbor_type=1)
            filtered_tokens_jem = np.array(filtered_tokens_sorted, dtype=object)[joint_out]
            filtered_tokens_jem = [token_tuple[0][0] for token_tuple in filtered_tokens_jem]
            print(filtered_tokens_jem)
            
            # lowest ppl reference question
            paraphrase_sentences = []
            for t in TEMPERATURE_LIST:
                if len(paraphrased_df.loc[i][f"T_{t}"]) > 0:
                    paraphrase_sentences.append(paraphrased_df.loc[i][f"T_{t}"])
                else:
                    paraphrase_sentences.append(" ")
                    
            perplexity_res = perplexity_metric.compute(predictions=paraphrase_sentences, model_id="gpt2")
            tmp_df = pd.DataFrame({"Predictions": paraphrase_sentences, "Perplexity": perplexity_res['perplexities']})
            lowest_perplexity_idx = tmp_df["Perplexity"].idxmin()
            reference_question = tmp_df.loc[lowest_perplexity_idx]["Predictions"]
            # ================ End Here ================

            random.shuffle(filtered_tokens_jem)  
            random.shuffle(filtered_tokens_ndp)  
            
            # For Non-DP
            # Build tokens prompt
            suggest_tokens = ""
            for token in filtered_tokens_ndp:
                suggest_tokens += token + ", "
            suggest_tokens = suggest_tokens[:-2]

            # Build Prompt and generate questions
            icl_prompt = (
                "Refer the following question to generate a new question:\n"
                + reference_question
                + "\nAvoid using following tokens:\n"
                # + "\nDo not using following tokens:\n"
                + suggest_tokens
                + "\nGenerated question:\n"
            )
            question = model.generate(icl_prompt, reference_question)['output_text']
            question = model.clean_text(question, icl_prompt)

            ## cloud enable section ##
            # Generate answers
            prompt = ""
            if DATASET_TYPE == "medqa":
                for k,v in paraphrased_df.loc[i]['choices'].items():
                    prompt += k + ". " + v + '\n'
                prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            elif DATASET_TYPE == "csqa":
                for idx in range(len(paraphrased_df.loc[i]['choices']['label'])):
                    prompt += paraphrased_df.loc[i]['choices']['label'][idx] + ". " + paraphrased_df.loc[i]['choices']['text'][idx] + '\n'
                prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            
            answers = generate(prompt, temperature=0.0, logits_dict=LOGITS_BIAS_DICT, max_tokens=1)

            questions_list.append(question)
            answers_list.append(answers)
            
            if len(paraphrased_df.loc[i]['reference_question']) == 0:
                paraphrased_df.loc[i, 'reference_question'] = reference_question
            paraphrased_df.loc[i, 'ndp_tokens'].append(filtered_tokens_ndp)
            paraphrased_df.loc[i, 'ndp_question'].append(question)
            paraphrased_df.loc[i, 'ndp_answer'].append(answers)
            
            # For JointEM
            # Build tokens prompt
            suggest_tokens = ""
            for token in filtered_tokens_jem:
                suggest_tokens += token + ", "
            suggest_tokens = suggest_tokens[:-2]
            
            # Build Prompt and generate questions
            icl_prompt = (
                "Refer the following question to generate a new question:\n"
                + reference_question
                + "\nAvoid using following tokens:\n"
                # + "\nDo not using following tokens:\n"
                + suggest_tokens
                + "\nGenerated question:\n"
            )
            question = model.generate(icl_prompt, reference_question)['output_text']
            question = model.clean_text(question, icl_prompt)
            
            ## cloud enable section ##
            # Generate answers
            prompt = ""
            if DATASET_TYPE == "medqa":
                for k,v in paraphrased_df.loc[i]['choices'].items():
                    prompt += k + ". " + v + '\n'
                prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            elif DATASET_TYPE == "csqa":
                for idx in range(len(paraphrased_df.loc[i]['choices']['label'])):
                    prompt += paraphrased_df.loc[i]['choices']['label'][idx] + ". " + paraphrased_df.loc[i]['choices']['text'][idx] + '\n'
                prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            
            answers = generate(prompt, temperature=0.0, logits_dict=LOGITS_BIAS_DICT, max_tokens=1)
            
            jem_questions_list.append(question)
            jem_answers_list.append(answers)
            
            paraphrased_df.loc[i, 'jem_tokens'].append(filtered_tokens_jem)
            paraphrased_df.loc[i, 'jem_question'].append(question)
            paraphrased_df.loc[i, 'jem_answer'].append(answers)
            
        try:
            bleu_q_score = bleu_metric.compute(references=questions_gt_list, predictions=questions_list)
        except:
            bleu_q_score = {'bleu': 0}
        try:
            rouge_q_score = rouge_metric.compute(references=questions_gt_list, predictions=questions_list)
        except:
            rouge_q_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        answers_list = convert_ABCDE(answers_list)
        acc_score = acc_metric.compute(references=answers_gt_list, predictions=answers_list)
        bleu_q_scores.append(bleu_q_score["bleu"])
        rouge1_q_scores.append(rouge_q_score["rouge1"])
        rouge2_q_scores.append(rouge_q_score["rouge2"])
        rougeL_q_scores.append(rouge_q_score["rougeL"])
        rougeLSum_q_scores.append(rouge_q_score["rougeLsum"])
        acc_scores.append(acc_score['accuracy'])
        
        try:
            bleu_q_jem_score = bleu_metric.compute(references=questions_gt_list, predictions=jem_questions_list)
        except:
            bleu_q_jem_score = {'bleu': 0}
        try:
            rouge_q_jem_score = rouge_metric.compute(references=questions_gt_list, predictions=jem_questions_list)
        except:
            rouge_q_jem_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLSum': 0}
        jem_answers_list = convert_ABCDE(jem_answers_list)
        acc_jem_score = acc_metric.compute(references=answers_gt_list, predictions=jem_answers_list)
        bleu_q_jointEM_scores.append(bleu_q_jem_score["bleu"])
        rouge1_q_jointEM_scores.append(rouge_q_jem_score["rouge1"])
        rouge2_q_jointEM_scores.append(rouge_q_jem_score["rouge2"])
        rougeL_q_jointEM_scores.append(rouge_q_jem_score["rougeL"])
        rougeLSum_q_jointEM_scores.append(rouge_q_jem_score["rougeLsum"])
        acc_jointEM_scores.append(acc_jem_score['accuracy'])

    append2file(res_path, f"Non-DP Question generated:")
    append2file(res_path, f"rouge1 mean: {np.mean(rouge1_q_scores)}; rouge2 mean: {np.mean(rouge2_q_scores)}; rougeL mean: {np.mean(rougeL_q_scores)}; rougeLSum mean: {np.mean(rougeLSum_q_scores)}")
    append2file(res_path, f"rouge1 std: {np.std(rouge1_q_scores)}; rouge2 std: {np.std(rouge2_q_scores)}; rougeL std: {np.std(rougeL_q_scores)}; rougeLSum std: {np.std(rougeLSum_q_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_q_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_q_scores)}")
    append2file(res_path, f"Answer Accuracy mean: {np.mean(acc_scores)}")
    append2file(res_path, f"Answer Accuracy std: {np.std(acc_scores)}")
    append2file(res_path, "<" * 50)
    
    append2file(res_path, f"JointEM Question generated:")
    append2file(res_path, f"rouge1 mean: {np.mean(rouge1_q_jointEM_scores)}; rouge2 mean: {np.mean(rouge2_q_jointEM_scores)}; rougeL mean: {np.mean(rougeL_q_jointEM_scores)}; rougeLSum mean: {np.mean(rougeLSum_q_jointEM_scores)}")
    append2file(res_path, f"rouge1 std: {np.std(rouge1_q_jointEM_scores)}; rouge2 std: {np.std(rouge2_q_jointEM_scores)}; rougeL std: {np.std(rougeL_q_jointEM_scores)}; rougeLSum std: {np.std(rougeLSum_q_jointEM_scores)}")
    append2file(res_path, f"bleu mean: {np.mean(bleu_q_jointEM_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_q_jointEM_scores)}")
    append2file(res_path, f"Answer Accuracy mean: {np.mean(acc_jointEM_scores)}")
    append2file(res_path, f"Answer Accuracy std: {np.std(acc_jointEM_scores)}")
    append2file(res_path, "=" * 50)
    
    paraphrased_df.to_json(f"{path_prefix}fin_T{paraphrase_T}.json", orient="records")


# ------------------ 主程序入口 ------------------
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run GPT35 experiments for csQA, medQA, or VQA")
    # parser.add_argument("--dataset", type=str, required=True, choices=["csQA", "medQA", "VQA"],
    #                     help="Dataset type to run (csQA, medQA, VQA)")
    # parser.add_argument("--paraphrase_T", type=float, default=0,
    #                     help="Paraphrase T value (e.g. 0, 0.25, etc.)")
    # args = parser.parse_args()

    # dataset_list = ["csQA", "medQA", "VQA"]
    dataset_list = ["medQA", "VQA"]
    paraphrase_T_list = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    for dataset in dataset_list:
        DATASET_TYPE = dataset
        for paraphrase_T in paraphrase_T_list:
            if dataset == "csQA":
                run_csqa(paraphrase_T)
            elif dataset == "medQA":
                run_medqa(paraphrase_T)
            elif dataset == "VQA":
                run_vqa(paraphrase_T)
            else:
                print("Invalid dataset type!")
            torch.cuda.empty_cache()
            gc.collect()
