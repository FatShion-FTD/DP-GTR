from utils import append2file
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os
from dpps.LLMDP import DPParaphrase
from dpps.DPMLM import DPMLM
from dpps.SLM import SLM
import gc
import torch
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
import random
from scipy import stats
import logging, sys
log_format = '%(asctime)s %(message)s'  
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(filename='log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

data = load_dataset("nielsr/docvqa_1200_examples_donut")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
perplexity_metric = evaluate.load("perplexity", module_type="metric")
TEST_SIZE = 200     # TEST = 10, ALL = 200
repeats = 3


def slm_vqa_exp(model:SLM, epsilon_list):
    for e_idx, epsilon in enumerate(epsilon_list):
        print("Old Config:", model.get_config())
        model.clip_model(epsilon=epsilon, clip_type="all_clip")
        print("New Config:", model.get_config())
        path_prefix = "results/VQA_res/"
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)

        res_path = f"{path_prefix}res_e{epsilon}.txt"
        append2file(res_path, "\nPTR Avoid lowest ppl reference ICL Experiment:")
        append2file(res_path, f"Config: {model.get_config()}")
        append2file(res_path, f"{model.model_name} S1 Experiment:")

        test_df = data["test"].to_pandas()
        test_df = test_df.head(TEST_SIZE)
        
        def question_paraphrase(question):
            prompt = f"Question : {question}\nParaphrase of the question :"
            output = model.generate(prompt, prompt)['output_text']
            return model.clean_text(output, prompt)

        # Paraphrase S1 question
        paraphrased_df = pd.DataFrame()
        temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        for tempreture in temperture_list:
            paraphrased_df[f"T_{tempreture}"] = test_df["query"].apply(
                lambda x: question_paraphrase(x["en"])
            )
            
        
        # Compute S1 Question
        references = test_df["query"].apply(lambda x: x["en"]).tolist()
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        bleu_scores = []
        paraphrased_df.fillna(" ")
        for tempreture in temperture_list:
            predictions = paraphrased_df[f"T_{tempreture}"].tolist()
            try:
                score = rouge_metric.compute(references=references, predictions=predictions)
            except:
                score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            rouge1_scores.append(score["rouge1"])
            rouge2_scores.append(score["rouge2"])
            rougeL_scores.append(score["rougeL"])
            try:
                bleu_score = bleu_metric.compute(references=references, predictions=predictions)
            except:
                bleu_score = {"bleu": 0}
            bleu_scores.append(bleu_score["bleu"])

        append2file(res_path, f"Question paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; rougeL mean: {np.mean(rougeL_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; rougeL std: {np.std(rougeL_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

        # Compute S1 Answer
        gt_answer = []
        for i in range(len(test_df)):
            answer = ""
            for word in test_df.loc[i]["answers"]:
                answer += word + ", "
            answer = answer[:-2]
            gt_answer.append(answer)

        rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

        for t in temperture_list:
            T_predictions = []
            for i in range(len(test_df)):
                prompt = ""
                for word in test_df.loc[i]["words"]:
                    prompt += word + ", "
                prompt = (
                    "Extracted OCR tokens from image:\n"
                    + prompt[:-2]
                    + "\nQuestion: "
                    + paraphrased_df.loc[i][f"T_{t}"]
                    + "\nAnswer the question with short term:\n"
                )
                T_predictions.append(generate(prompt, temperature=0.0, stop=["\n"]))
            try:
                score = rouge_metric.compute(references=gt_answer, predictions=T_predictions)
            except:
                score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            rouge1_scores.append(score["rouge1"])
            rouge2_scores.append(score["rouge2"])
            rougeL_scores.append(score["rougeL"])
            try:
                bleu_score = bleu_metric.compute(references=gt_answer, predictions=T_predictions)
            except:
                bleu_score = {"bleu": 0}
            bleu_scores.append(bleu_score["bleu"])

        append2file(res_path, f"Answer paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; rougeL mean: {np.mean(rougeL_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; rougeL std: {np.std(rougeL_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

        # Concat the paraphrased questions with the original questions and answers
        paraphrased_df["original_question"] = test_df["query"].apply(lambda x: x["en"])
        paraphrased_df["original_answer"] = test_df["answers"]
        paraphrased_df["words"] = test_df["words"]

        df_file_path = f"{path_prefix}{model.model_name}_JSON/paraphrased_questions_e{epsilon}.json"
        
        if not os.path.exists(f"{path_prefix}{model.model_name}_JSON/"):
            os.makedirs(f"{path_prefix}{model.model_name}_JSON/")

        # Save the paraphrased questions
        paraphrased_df.to_json(df_file_path, orient="records")


        # Step2: KSA Generate Questions 
        paraphrased_df = pd.read_json(df_file_path)
        # paraphrased_df.head()
        questions_gt_list = paraphrased_df["original_question"].tolist()
        answers_gt_list = paraphrased_df["original_answer"].tolist()
        paraphrased_df = paraphrased_df.fillna(" ")

        stopword_set = set(stopwords.words("english"))
        temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        (
            bleu_q_scores,
            bleu_a_scores,
            rouge1_q_scores,
            rouge2_q_scores,
            rougeL_q_scores,
            rouge1_a_scores,
            rouge2_a_scores,
            rougeL_a_scores,
        ) = ([], [], [], [], [], [], [], [])
        
        analyt_df = pd.DataFrame()
        analyt_df["OQ"] = paraphrased_df["original_question"]
        for idx, t in enumerate(temperture_list):
            analyt_df[f"P_{idx}"] = paraphrased_df[f"T_{t}"]
        analyt_df_save_path = f"{path_prefix}{model.model_name}_JSON/analyt_df_e{epsilon}.json"

        for r in range(repeats):
            questions_list = []
            answers_list = []
            reference_question_list = []
            generated_question_list = []
            
            for i in range(len(paraphrased_df)):
                # Count token freq
                all_tokens = {}  # key: token, value: count
                for t in temperture_list:
                    sentence = paraphrased_df.loc[i][f"T_{t}"]
                    tokens = nltk.word_tokenize(sentence)
                    onegrams = set(ngrams(tokens, 1))
                    # onegrams = set(onegrams)
                    # making onegrams a set to avoid duplicate tokens
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

                
                # ========== PTR =============
                # 1) 计算 gap_lst
                try:
                    gap_lst = []
                    max_k = min(len(filtered_tokens_sorted)-1, 30)  # 最多考虑 30
                    for k in range(max_k):
                        gap_k = filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k+1][1]
                        gap_lst.append(gap_k)

                    if len(filtered_tokens_sorted) <= 30:
                        gap_lst.append(filtered_tokens_sorted[-1][1])

                    gap_lst = np.array(gap_lst)
                    kstar = np.argmax(gap_lst)
                    dk = gap_lst[kstar]

                    # 2) 加高斯噪声
                    privacy_params = {
                    'sigma': 1.0,
                    'delta0': 1e-5
                    }
                    noise1 = np.random.normal(0, 2*privacy_params['sigma'])
                    noise2 = stats.norm.isf(privacy_params['delta0'], loc=0, scale=2*privacy_params['sigma'])
                    dkhat = max(2, dk) + noise1 - noise2
                    print(f"dk={dk}, noise1={noise1}, noise2={noise2}, dkhat={dkhat}")

                    # 3) PTR 决策
                    #   - 这里 "through" 就表示“可发布(avoid)前 kstar+1 个 tokens”
                    #   - 这里 "fail" 就表示不要把这些 tokens 放进 avoid list

                    if dkhat > 2:
                        print(f"***PASS TEST! RELEASE EXACT TOP-{kstar+1} TOKENS***")
                        topk = kstar + 1
                        # 再跟 25% 逻辑结合
                        top25 = int(len(filtered_tokens_sorted)*0.25)
                        topk = min(topk, top25)
                        # 取前 topk
                        filtered_tokens_sorted = filtered_tokens_sorted[:topk]
                        filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
                    else:
                        print("***FAIL TEST! DO Zero-shot or Publish Empty Tokens***")
                        logging.info(f" {i}-th question has {dkhat} noise under {model.model_name}, empty tokens")
                        filtered_tokens = []
                except:
                    logging.info(f" {i}-th question has empty tokens under {model.model_name}")
                    filtered_tokens = []

                # ================ End Here ================

                random.shuffle(filtered_tokens)  # shuffle the list of tokens

                # Build tokens prompt
                suggest_tokens = ""
                for token in filtered_tokens:
                    suggest_tokens += token + ", "
                suggest_tokens = suggest_tokens[:-2]
                
                # lowest ppl reference question
                paraphrase_sentences = []
                for t in temperture_list:
                    if len(paraphrased_df.loc[i][f"T_{t}"]) > 0:
                        paraphrase_sentences.append(paraphrased_df.loc[i][f"T_{t}"])
                    else:
                        paraphrase_sentences.append(" ")
                        
                perplexity_res = perplexity_metric.compute(predictions=paraphrase_sentences, model_id="gpt2")
                tmp_df = pd.DataFrame({"Predictions": paraphrase_sentences, "Perplexity": perplexity_res['perplexities']})
                lowest_perplexity_idx = tmp_df["Perplexity"].idxmin()
                reference_question = tmp_df.loc[lowest_perplexity_idx]["Predictions"]

                # Build Prompt and generate questions
                icl_prompt = (
                        "Paraphrase the following question:\n"
                        + reference_question
                        + "\nAvoid using following tokens:\n"
                        + suggest_tokens
                        + "\nParaphrased question:"
                    )
                question = model.generate(icl_prompt, reference_question)['output_text']
                question = model.clean_text(question, icl_prompt)
                
                reference_question_list.append(reference_question)
                generated_question_list.append(question)

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
            
            analyt_df[f"R{r}_RQ"] = reference_question_list
            analyt_df[f"R{r}_GQ"] = generated_question_list
            answers_list = pd.Series(answers_list).fillna(" ").tolist()
            questions_list = pd.Series(questions_list).fillna(" ").tolist()
            
            try:
                bleu_q_score = bleu_metric.compute(references=questions_gt_list, predictions=questions_list)
            except:
                bleu_q_score = {"bleu": 0}
            
            try:
                rouge_q_score = rouge_metric.compute(references=questions_gt_list, predictions=questions_list)
            except:
                rouge_q_score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
                
            try:
                bleu_ans_score = bleu_metric.compute(references=answers_gt_list, predictions=answers_list)
            except:
                bleu_ans_score = {"bleu": 0}
            
            try:
                rouge_ans_score = rouge_metric.compute(references=answers_gt_list, predictions=answers_list)
            except:
                rouge_ans_score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            
            bleu_q_scores.append(bleu_q_score["bleu"])
            rouge1_q_scores.append(rouge_q_score["rouge1"])
            rouge2_q_scores.append(rouge_q_score["rouge2"])
            rougeL_q_scores.append(rouge_q_score["rougeL"])
            bleu_a_scores.append(bleu_ans_score["bleu"])
            rouge1_a_scores.append(rouge_ans_score["rouge1"])
            rouge2_a_scores.append(rouge_ans_score["rouge2"])
            rougeL_a_scores.append(rouge_ans_score["rougeL"])

        save_rq = True
        for r in range(repeats):
            if save_rq:
                analyt_df["RQ"] = analyt_df[f"R{r}_RQ"]
                save_rq = False
            analyt_df = analyt_df.drop(columns=[f"R{r}_RQ"])

        analyt_df.to_json(analyt_df_save_path, orient="records")
        
        append2file(res_path, f"S2 Question generated:")
        append2file(
            res_path,
            f"BLEU MEAN Question: {np.mean(bleu_q_scores)} STD Question: {np.std(bleu_q_scores)}",
        )
        append2file(
            res_path,
            f"ROUGE1 MEAN Question: {np.mean(rouge1_q_scores)} STD Question: {np.std(rouge1_q_scores)}",
        )
        append2file(
            res_path,
            f"ROUGE2 MEAN Question: {np.mean(rouge2_q_scores)} STD Question: {np.std(rouge2_q_scores)}",
        )
        append2file(
            res_path,
            f"ROUGEL MEAN Question: {np.mean(rougeL_q_scores)} STD Question: {np.std(rougeL_q_scores)}",
        )

        append2file(res_path, f"S2 Answer generated:")
        append2file(
            res_path,
            f"BLEU MEAN Answer: {np.mean(bleu_a_scores)} STD Answer: {np.std(bleu_a_scores)}",
        )
        append2file(
            res_path,
            f"ROUGE1 MEAN Answer: {np.mean(rouge1_a_scores)} STD Answer: {np.std(rouge1_a_scores)}",
        )
        append2file(
            res_path,
            f"ROUGE2 MEAN Answer: {np.mean(rouge2_a_scores)} STD Answer: {np.std(rouge2_a_scores)}",
        )
        append2file(
            res_path,
            f"ROUGEL MEAN Answer: {np.mean(rougeL_a_scores)} STD Answer: {np.std(rougeL_a_scores)}",
        )
        gc.collect()


if __name__ == "__main__":
    epsilon_list = [200, 150, 100, 75, 50, 25, 10, 5]
    model_name_list = [
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "meta-llama/Llama-3.2-1B",
        "EleutherAI/gpt-neo-1.3B",
        "HuggingFaceTB/SmolLM-1.7B",
        "facebook/opt-1.3b",
        
        # "Qwen/Qwen2-1.5B",
        # "meta-llama/Llama-3.2-3B",
        # "EleutherAI/pythia-1.4b",
        # "ahxt/LiteLlama-460M-1T",
    ]
    
    for model_name in model_name_list:
        model = SLM(model_name)
        slm_vqa_exp(model, epsilon_list)
        torch.cuda.empty_cache()