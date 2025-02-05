import argparse
from utils import append2file, str_to_list, convert_ABCDE, create_choices
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os, sys
from dpps.jointEM import joint
from dpps.DPMLM import DPMLM
from dpps.LLMDP import DPParaphrase
from dpps.HaS import HaS
import torch

# 全局变量（将在main中通过参数赋值）
TEST_SIZE = None
DATASET_TYPE = None  # 可选：'csQA', 'medQA', 'VQA'
EXP_TYPE = None      # 可选：'DPMLM', 'DPParaphrase', 'HAS'

temperture_list = [1,2,3,4,5,6,7,8,9,10]
logits_bias_dict = {"32": 100, "33": 100, "34": 100, "35": 100, "36": 100}     # A-E: 32-35

# path_prefix 依赖 DATASET_TYPE 和 EXP_TYPE，稍后在main中初始化
path_prefix = None

paraphrase_Ts = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# EXP_NAME 中使用 format 格式化
EXP_NAME = "\nBaseline {EXP_TYPE} on {DATASET_TYPE} Experiment:"

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
acc_metric = evaluate.load("accuracy")
perplexity_metric = evaluate.load("perplexity", module_type="metric")

def build_prompt(df, i, t):
    if DATASET_TYPE == 'csQA':
        prompt = ""
        for idx in range(len(df.loc[i]['choices']['label'])):
            prompt += df.loc[i]['choices']['label'][idx] + ". " + df.loc[i]['choices']['text'][idx] + '\n'
        prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
        return prompt
    elif DATASET_TYPE == 'medQA':
        prompt = ""
        for k, v in df.loc[i]['choices'].items():
            prompt += k + ". " + v + '\n'
        prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
        return prompt
    elif DATASET_TYPE == 'VQA':
        prompt = ""
        if EXP_TYPE == 'HAS':
            prompt = df.loc[i]["words"]
        else:
            for word in df.loc[i]["words"]:
                prompt += word + ", "
            prompt = prompt[:-2]
                
        prompt = (
            "Extracted OCR tokens from image:\n"
            + prompt
            + "\nQuestion: "
            + df.loc[i][f"T_{t}"]
            + "\nAnswer the question with short term:\n"
        )
        return prompt

def get_df():
    paraphrased_df = pd.DataFrame()

    if DATASET_TYPE == 'csQA':
        data = load_dataset("tau/commonsense_qa")
        test_df = data['validation'].to_pandas()
        test_df = test_df.head(TEST_SIZE)
        paraphrased_df['original_question'] = test_df['question']
        paraphrased_df["original_answer"] = test_df["answerKey"]
        paraphrased_df['choices'] = test_df['choices']
    elif DATASET_TYPE == 'medQA':
        test_df = pd.read_json("dataset/medQA_4.json", orient="records")
        test_df = test_df.head(TEST_SIZE)
        paraphrased_df['original_question'] = test_df['question']
        paraphrased_df["original_answer"] = test_df["answer_idx"]
        paraphrased_df['choices'] = test_df['options']
    elif DATASET_TYPE == 'VQA':
        data = load_dataset("nielsr/docvqa_1200_examples_donut")
        test_df = data["test"].to_pandas()
        test_df = test_df.head(TEST_SIZE)
        paraphrased_df['original_question'] = test_df['query'].apply(lambda x: x['en'])
        paraphrased_df["original_answer"] = test_df["answers"]
        paraphrased_df['words'] = test_df['words']
    return paraphrased_df
    
def run_DPParaphrase():
    paraphrased_df = get_df()
    paraphrase_epss = [2 * 88 / xi for xi in paraphrase_Ts]
    print(paraphrase_epss)
    
    dp_para = DPParaphrase()
    for idx, eps in enumerate(paraphrase_epss):
        paraphrase_T = paraphrase_Ts[idx]
        res_path = f"{path_prefix}res_{paraphrase_T}_{eps}.txt"
        append2file(res_path, EXP_NAME.format(EXP_TYPE=EXP_TYPE, DATASET_TYPE=DATASET_TYPE))
        append2file(res_path, f"Paraphrase T: {paraphrase_T}")

        for tempreture in temperture_list:
            paraphrased_df[f"T_{tempreture}"] = paraphrased_df['original_question'].apply(
                lambda x: dp_para.privatize(
                    text=x,
                    epsilon=eps,
                )
            )
            
        # 检查每行所有温度下生成的文本是否均为空
        for i in range(len(paraphrased_df)):
            tot_len = 0
            for t in temperture_list:
                tot_len += len(paraphrased_df.loc[i][f"T_{t}"])
            if tot_len == 0:
                paraphrased_df.drop(i, inplace=True)
                append2file(res_path, f"row {i}: empty!!")
                
        # 对生成的改写问题计算各项指标
        references = paraphrased_df["original_question"].tolist()
        rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
        bleu_scores = []
        for tempreture in temperture_list:
            predictions = paraphrased_df[f"T_{tempreture}"].tolist()
            score = rouge_metric.compute(references=references, predictions=predictions)
            print(
                f"rouge1: {score['rouge1']}; rouge2: {score['rouge2']}; rougeL: {score['rougeL']}"
            )
            rouge1_scores.append(score["rouge1"])
            rouge2_scores.append(score["rouge2"])
            rougeL_scores.append(score["rougeL"])
            rougeLSum_scores.append(score["rougeLsum"])
            bleu_score = bleu_metric.compute(references=references, predictions=predictions)
            bleu_scores.append(bleu_score["bleu"])
            print(f"bleu: {bleu_score}")

        append2file(res_path, "Question paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
            f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
            f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_scores)}")
        df_file_path = f"{path_prefix}paraphrased_questions_T{paraphrase_T}.json"
        paraphrased_df.to_json(df_file_path, orient="records")
        
        if DATASET_TYPE in ['csQA', 'medQA']:
            gt_answer = paraphrased_df["original_answer"].tolist()
            gt_answer = convert_ABCDE(gt_answer)
            
            acc_scores = []
            for t in temperture_list:
                T_predictions = []
                for i in range(len(paraphrased_df)):
                    prompt = build_prompt(paraphrased_df, i, t)
                    T_predictions.append(generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1))
                T_predictions = convert_ABCDE(T_predictions)
                acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
                acc_scores.append(acc_score['accuracy'])
            append2file(res_path, "Answer paraphrased S1:")
            append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
            append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
            append2file(res_path, ">" * 50)
            
        else:
            gt_answer = []
            for i in range(len(paraphrased_df)):
                answer = ""
                for word in paraphrased_df.loc[i]["original_answer"]:
                    answer += word + ", "
                answer = answer[:-2]
                gt_answer.append(answer)
            print(gt_answer)
        
            rouge1_a_scores, rouge2_a_scores, rougeL_a_scores, rougeLSum_a_scores = [], [], [], []
            bleu_a_scores = []
            
            for t in temperture_list:
                T_predictions = []
                for i in range(len(paraphrased_df)):
                    prompt = build_prompt(paraphrased_df, i, t)
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

            append2file(res_path, "Answer paraphrased S1:")
            append2file(
                res_path,
                f"rouge1 mean: {np.mean(rouge1_a_scores)}; rouge2 mean: {np.mean(rouge2_a_scores)}; "
                f"rougeL mean: {np.mean(rougeL_a_scores)}; rougeLSum mean: {np.mean(rougeLSum_a_scores)}",
            )
            append2file(
                res_path,
                f"rouge1 std: {np.std(rouge1_a_scores)}; rouge2 std: {np.std(rouge2_a_scores)}; "
                f"rougeL std: {np.std(rougeL_a_scores)}; rougeLSum std: {np.std(rougeLSum_a_scores)}",
            )
            append2file(res_path, f"bleu mean: {np.mean(bleu_a_scores)}")
            append2file(res_path, f"bleu std: {np.std(bleu_a_scores)}")
            append2file(res_path, ">" * 50)
    
    
def run_DPMLM():
    paraphrased_df = get_df()
    paraphrase_epss = [2 * 19.5 / xi for xi in paraphrase_Ts]
    print(paraphrase_epss)
    
    dpmlm = DPMLM()
    for idx, eps in enumerate(paraphrase_epss):
        paraphrase_T = paraphrase_Ts[idx]
        res_path = f"{path_prefix}res_{paraphrase_T}_{eps}.txt"
        append2file(res_path, EXP_NAME.format(EXP_TYPE=EXP_TYPE, DATASET_TYPE=DATASET_TYPE))
        append2file(res_path, f"Paraphrase T: {paraphrase_T}")

        for tempreture in temperture_list:
            paraphrased_df[f"T_{tempreture}"] = paraphrased_df['original_question'].apply(
                lambda x: dpmlm.dpmlm_rewrite(x, eps, REPLACE=True, FILTER=True, STOP=True, TEMP=True, POS=True, CONCAT=True)
            )
            
        for i in range(len(paraphrased_df)):
            tot_len = 0
            for t in temperture_list:
                tot_len += len(paraphrased_df.loc[i][f"T_{t}"])
            if tot_len == 0:
                paraphrased_df.drop(i, inplace=True)
                append2file(res_path, f"row {i}: empty!!")
                
        references = paraphrased_df["original_question"].tolist()
        rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
        bleu_scores = []
        for tempreture in temperture_list:
            predictions = paraphrased_df[f"T_{tempreture}"].tolist()
            score = rouge_metric.compute(references=references, predictions=predictions)
            print(
                f"rouge1: {score['rouge1']}; rouge2: {score['rouge2']}; rougeL: {score['rougeL']}"
            )
            rouge1_scores.append(score["rouge1"])
            rouge2_scores.append(score["rouge2"])
            rougeL_scores.append(score["rougeL"])
            rougeLSum_scores.append(score["rougeLsum"])
            bleu_score = bleu_metric.compute(references=references, predictions=predictions)
            bleu_scores.append(bleu_score["bleu"])
            print(f"bleu: {bleu_score}")

        append2file(res_path, "Question paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
            f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
            f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_scores)}")
        df_file_path = f"{path_prefix}paraphrased_questions_T{paraphrase_T}.json"
        paraphrased_df.to_json(df_file_path, orient="records")
        
        if DATASET_TYPE in ['csQA', 'medQA']:
            gt_answer = paraphrased_df["original_answer"].tolist()
            gt_answer = convert_ABCDE(gt_answer)
            
            acc_scores = []
            for t in temperture_list:
                T_predictions = []
                for i in range(len(paraphrased_df)):
                    prompt = build_prompt(paraphrased_df, i, t)
                    T_predictions.append(generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1))
                T_predictions = convert_ABCDE(T_predictions)
                acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
                acc_scores.append(acc_score['accuracy'])
            append2file(res_path, "Answer paraphrased S1:")
            append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
            append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
            append2file(res_path, ">" * 50)
            
        else:
            gt_answer = []
            for i in range(len(paraphrased_df)):
                answer = ""
                for word in paraphrased_df.loc[i]["original_answer"]:
                    answer += word + ", "
                answer = answer[:-2]
                gt_answer.append(answer)
            print(gt_answer)
        
            rouge1_a_scores, rouge2_a_scores, rougeL_a_scores, rougeLSum_a_scores = [], [], [], []
            bleu_a_scores = []
            
            for t in temperture_list:
                T_predictions = []
                for i in range(len(paraphrased_df)):
                    prompt = build_prompt(paraphrased_df, i, t)
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

            append2file(res_path, "Answer paraphrased S1:")
            append2file(
                res_path,
                f"rouge1 mean: {np.mean(rouge1_a_scores)}; rouge2 mean: {np.mean(rouge2_a_scores)}; "
                f"rougeL mean: {np.mean(rougeL_a_scores)}; rougeLSum mean: {np.mean(rougeLSum_a_scores)}",
            )
            append2file(
                res_path,
                f"rouge1 std: {np.std(rouge1_a_scores)}; rouge2 std: {np.std(rouge2_a_scores)}; "
                f"rougeL std: {np.std(rougeL_a_scores)}; rougeLSum std: {np.std(rougeLSum_a_scores)}",
            )
            append2file(res_path, f"bleu mean: {np.mean(bleu_a_scores)}")
            append2file(res_path, f"bleu std: {np.std(bleu_a_scores)}")
            append2file(res_path, ">" * 50)
    
def run_HAS():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    has = HaS(device)
    res_path = f"{path_prefix}res.txt"
    append2file(res_path, EXP_NAME.format(EXP_TYPE=EXP_TYPE, DATASET_TYPE=DATASET_TYPE))
    paraphrased_df = get_df()

    for tempreture in temperture_list:
        paraphrased_df[f"T_{tempreture}"] = paraphrased_df['original_question'].apply(
            lambda x: has.hide(x)
        )
        
    def hide_choices(x):
        hidden_choices_dict = {}
        for k, v in x.items():
            hidden_choices_dict[k] = has.hide(v)
        print(f"hidden_choices_dict: {hidden_choices_dict}")
        return hidden_choices_dict
        
    def hide_words(x):
        hidden_words = ""
        for word in x:
            hidden_words += word + ", "
        hidden_words = has.hide(hidden_words[:-2])
        print(f"hidden_words: {hidden_words}")
        return hidden_words
        
    if DATASET_TYPE in ['csQA', 'medQA']:
        paraphrased_df['choices'] = paraphrased_df['choices'].apply(lambda x: hide_choices(x))
    else:
        paraphrased_df['words'] = paraphrased_df['words'].apply(lambda x: hide_words(x))
        
    for i in range(len(paraphrased_df)):
        tot_len = 0
        for t in temperture_list:
            tot_len += len(paraphrased_df.loc[i][f"T_{t}"])
        if tot_len == 0:
            paraphrased_df.drop(i, inplace=True)
            append2file(res_path, f"row {i}: empty!!")
            
    references = paraphrased_df["original_question"].tolist()
    rouge1_scores, rouge2_scores, rougeL_scores, rougeLSum_scores = [], [], [], []
    bleu_scores = []
    for tempreture in temperture_list:
        predictions = paraphrased_df[f"T_{tempreture}"].tolist()
        score = rouge_metric.compute(references=references, predictions=predictions)
        print(
            f"rouge1: {score['rouge1']}; rouge2: {score['rouge2']}; rougeL: {score['rougeL']}"
        )
        rouge1_scores.append(score["rouge1"])
        rouge2_scores.append(score["rouge2"])
        rougeL_scores.append(score["rougeL"])
        rougeLSum_scores.append(score["rougeLsum"])
        bleu_score = bleu_metric.compute(references=references, predictions=predictions)
        bleu_scores.append(bleu_score["bleu"])
        print(f"bleu: {bleu_score}")

    append2file(res_path, "Question paraphrased S1:")
    append2file(
        res_path,
        f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; "
        f"rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}",
    )
    append2file(
        res_path,
        f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; "
        f"rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}",
    )
    append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
    append2file(res_path, f"bleu std: {np.std(bleu_scores)}")
    df_file_path = f"{path_prefix}paraphrased_questions.json"
    paraphrased_df.to_json(df_file_path, orient="records")
    
    if DATASET_TYPE in ['csQA', 'medQA']:
        gt_answer = paraphrased_df["original_answer"].tolist()
        gt_answer = convert_ABCDE(gt_answer)
        
        acc_scores = []
        for t in temperture_list:
            T_predictions = []
            for i in range(len(paraphrased_df)):
                prompt = build_prompt(paraphrased_df, i, t)
                T_predictions.append(generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1))
            T_predictions = convert_ABCDE(T_predictions)
            acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
            acc_scores.append(acc_score['accuracy'])
        append2file(res_path, "Answer paraphrased S1:")
        append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
        append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
        append2file(res_path, ">" * 50)
        
    else:
        gt_answer = []
        for i in range(len(paraphrased_df)):
            answer = ""
            for word in paraphrased_df.loc[i]["original_answer"]:
                answer += word + ", "
            answer = answer[:-2]
            gt_answer.append(answer)
        print(gt_answer)
    
        rouge1_a_scores, rouge2_a_scores, rougeL_a_scores, rougeLSum_a_scores = [], [], [], []
        bleu_a_scores = []
        
        for t in temperture_list:
            T_predictions = []
            for i in range(len(paraphrased_df)):
                prompt = build_prompt(paraphrased_df, i, t)
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

        append2file(res_path, "Answer paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_a_scores)}; rouge2 mean: {np.mean(rouge2_a_scores)}; "
            f"rougeL mean: {np.mean(rougeL_a_scores)}; rougeLSum mean: {np.mean(rougeLSum_a_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_a_scores)}; rouge2 std: {np.std(rouge2_a_scores)}; "
            f"rougeL std: {np.std(rougeL_a_scores)}; rougeLSum std: {np.std(rougeLSum_a_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_a_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_a_scores)}")
        append2file(res_path, ">" * 50)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DP Paraphrase Experiments.")
    parser.add_argument("--test_size", type=int, default=10, help="Number of test samples")
    parser.add_argument("--dataset_type", type=str, default="csQA", choices=["csQA", "medQA", "VQA"],
                        help="Dataset type to use")
    parser.add_argument("--exp_type", type=str, default="HAS", choices=["DPMLM", "DPParaphrase", "HAS"],
                        help="Experiment type to run")
    args = parser.parse_args()

    # 通过命令行参数赋值全局变量
    TEST_SIZE = args.test_size
    DATASET_TYPE = args.dataset_type
    EXP_TYPE = args.exp_type

    # 根据 DATASET_TYPE 和 EXP_TYPE 构造结果输出路径
    path_prefix = f"results/{DATASET_TYPE}/{EXP_TYPE}/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    if EXP_TYPE == 'DPMLM':
        run_DPMLM()
    elif EXP_TYPE == 'DPParaphrase':
        run_DPParaphrase()
    elif EXP_TYPE == 'HAS':
        run_HAS()
    else:
        print('Invalid EXP_TYPE')
        sys.exit(0)
