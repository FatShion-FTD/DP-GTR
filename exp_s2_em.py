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

def step2_vqa(paraphrased_df:pd.DataFrame, res_path, path_prefix, paraphrase_T):
    questions_gt_list = paraphrased_df["original_question"].tolist()
    answers_gt_list = paraphrased_df["original_answer"].tolist()
    paraphrased_df = paraphrased_df.fillna(" ")
    
    # unclip model
    model.unclip_model()
    
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
    
    # unclip model
    model.unclip_model()
    
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






def process_paraphrased_questions(folder):      # folder: results/VQA/SLM_Llama8B/
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
        file_order = float(file_order)
        
        # 使用 pandas 读取 json 文件，返回 DataFrame 格式
        try:
            df = pd.read_json(file_path)
            # ================================ Test ================================
            # df = df.head(2)
            # ================================ Test ================================
        except ValueError as e:
            print(f"读取 {file_path} 出现错误: {e}")
            continue
        res_path = os.path.join(folder, f"res_{file_order}.txt")
        if 'VQA' in folder:
            DATASET_TYPE = "VQA"
            step2_vqa(df, res_path, path_prefix=f"{folder}s2_", paraphrase_T=file_order)
        else:
            if 'csQA' in folder:
                DATASET_TYPE = "csQA"
            elif 'medQA' in folder:
                DATASET_TYPE = "medQA"
            step2_csQA_medQA(df, res_path, path_prefix=f"{folder}s2_", paraphrase_T=file_order)
    


# ================================ Test ================================
# paths = ['results/VQA/SLM_Llama8B/', 'results/csQA/SLM_Llama8B/']#, 'results/medQA/SLM_Llama8B/']
# ================================ Test ================================
paths = ['results/csQA/SLM_Llama8B/', 'results/VQA/SLM_Llama8B/', 'results/medQA/SLM_Llama8B/']


for path in paths:
    process_paraphrased_questions(path)
