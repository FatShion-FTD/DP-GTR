from utils import append2file, str_to_list
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os, sys
from utils import convert_ABCDE, create_choices
    
    
    
TEST_SIZE = 200
N = 2
dataset_path = f"dataset/commonsense_qa_{N}_choices.json"
if N == 5:
    data = load_dataset("tau/commonsense_qa")
    test_df = data['validation'].to_pandas()
else:
    test_df = pd.read_json(dataset_path, orient="records", lines=True)
test_df = test_df.head(TEST_SIZE)
    

def build_answer_prompt(df, i, t):
    prompt = ""
    for idx in range(len(df.loc[i]['choices']['label'])):
        prompt += df.loc[i]['choices']['label'][idx] + ". " + df.loc[i]['choices']['text'][idx] + '\n'
    prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
    return prompt


def run():
    paraphrase_Ts = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    logits_bias_dict = {"32": 100, "33": 100, "34": 100, "35": 100, "36": 100}     # A-E: 32-35
    path_prefix = "results/csQA_res/GPT35/"
    if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)



    for paraphrase_T in paraphrase_Ts:

        res_path = f"{path_prefix}res_{paraphrase_T}.txt"
        append2file(res_path, "\n25% Tokens Avoid Generation lowest ppl reference ICL Experiment:")
        append2file(res_path, f"Paraphrase T: {paraphrase_T}")

        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        acc_metric = evaluate.load("accuracy")

        paraphrased_df = pd.DataFrame()
        # Concat the paraphrased questions with the original questions and answers
        paraphrased_df['original_question'] = test_df['question']
        paraphrased_df["original_answer"] = test_df["answerKey"]
        paraphrased_df['choices'] = test_df['choices']
        
        
        # ## Step1.1: Paraphrase Questions
        temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        T = paraphrase_T
        for tempreture in temperture_list:
            paraphrased_df[f"T_{tempreture}"] = test_df["question"].apply(
                lambda x: generate(
                    f"Question : {x}\nParaphrase of the question :",
                    temperature=T,
                    stop=["\n"],
                )
            )

        references = test_df["question"].tolist()
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

        append2file(res_path, f"Question paraphrased S1:")
        append2file(
            res_path,
            f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; rougeL mean: {np.mean(rougeL_scores)}; rougeLSum mean: {np.mean(rougeLSum_scores)}",
        )
        append2file(
            res_path,
            f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; rougeL std: {np.std(rougeL_scores)}; rougeLSum std: {np.std(rougeLSum_scores)}",
        )
        append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
        append2file(res_path, f"bleu std: {np.std(bleu_scores)}")
        
        df_file_path = f"{path_prefix}paraphrased_questions_T{paraphrase_T}.json"

        # # Save the paraphrased questions
        paraphrased_df.to_json(df_file_path, orient="records")
        
        paraphrased_df = pd.read_json(df_file_path, orient="records")

        # N Choices
        # paraphrased_df = create_choices(paraphrased_df, 2)

        # # Step1.2: Generate Answers
        gt_answer = paraphrased_df["original_answer"].tolist()
        gt_answer = convert_ABCDE(gt_answer)
        repeats = 5
        
        acc_scores = []
        for t in temperture_list:
            T_predictions = []
            for i in range(len(paraphrased_df)):
                prompt = build_answer_prompt(paraphrased_df, i, t)

                T_predictions.append(generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1))
            T_predictions = convert_ABCDE(T_predictions)
            acc_score = acc_metric.compute(references=gt_answer, predictions=T_predictions)
            acc_scores.append(acc_score['accuracy'])

        append2file(res_path, f"Answer paraphrased S1:")
        append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")


        # Step2: KSA Generate Questions 
        # paraphrased_df.head()
        questions_gt_list = paraphrased_df["original_question"].tolist()
        answers_gt_list = paraphrased_df["original_answer"].tolist()
        answers_gt_list = convert_ABCDE(answers_gt_list)


        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        acc_metric = evaluate.load("accuracy")
        perplexity_metric = evaluate.load("perplexity", module_type="metric")
        
        import nltk
        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.util import ngrams

        stopword_set = set(stopwords.words("english"))
        import string
        import random

        temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        repeats = 5

        (
            bleu_q_scores,
            rouge1_q_scores,
            rouge2_q_scores,
            rougeL_q_scores,
            rougeLSum_q_scores
        ) = ([], [], [], [], [])
        
        acc_scores = []
        
        
        for _ in range(repeats):
            questions_list = []
            answers_list = []
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

                # Non-DP: find the count threshold where the count gap is the largest
                # actually_upper_bound = 0
                # count_threshold = (
                #     0  # you need to creat a list to store all difference between the counts
                # )
                # for k in range(len(filtered_tokens_sorted) - 1):
                #     # print(k,len(filtered_tokens_sorted)-2)
                #     if (
                #         filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k + 1][1]
                #         > count_threshold
                #     ):
                #         count_threshold = (
                #             filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k + 1][1]
                #         )
                #         actually_upper_bound = filtered_tokens_sorted[k][1]
                #     if k == len(filtered_tokens_sorted) - 2:
                #         if filtered_tokens_sorted[k + 1][1] > count_threshold:
                #             count_threshold = filtered_tokens_sorted[k + 1][1]
                #             actually_upper_bound = filtered_tokens_sorted[k + 1][
                #                 1
                #             ]  # including all tokens

                # # print(f"count_threshold: {count_threshold}")
                # filtered_tokens = dict(filtered_tokens_sorted)
                # filtered_tokens = [
                #     k[0] for k, v in filtered_tokens.items() if v >= actually_upper_bound
                # ]

                # Non-DP: Top 5 tokens
                # if len(filtered_tokens_sorted) > 5:
                #     filtered_tokens_sorted = filtered_tokens_sorted[:5]
                # filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
                # print(filtered_tokens)
                
                # Non-DP: 75% tokens, 50% tokens, 25% tokens
                filtered_tokens_sorted = filtered_tokens_sorted[: int(len(filtered_tokens_sorted) * 0.25)]
                filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
                print(filtered_tokens)
                
                # Non-DP: Find Intersect tokens
                # ori_tokens = nltk.word_tokenize(paraphrased_df.loc[i][f"original_question"])
                # filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
                # filtered_tokens = list(set(filtered_tokens).intersection(set(ori_tokens)))
                # print(filtered_tokens)
                
                # if len(filtered_tokens) == 0:
                #     print("No Intersect Tokens, take head 25% tokens")
                #     filtered_tokens_sorted = filtered_tokens_sorted[: int(len(filtered_tokens_sorted) * 0.25)]
                #     filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
                #     print(filtered_tokens)
                    

                # DP: RNM-Gaussian

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
                
                # reference_question = paraphrased_df.loc[i][f"original_question"]

                # Build Prompt and generate questions
                icl_prompt = (
                    "Refer the following question to generate a new question:\n"
                    + reference_question
                    + "\nAvoid using following tokens:\n"
                    # + "\nDo not using following tokens:\n"
                    + suggest_tokens
                    + "\nGenerated question:\n"
                )
                question = generate(icl_prompt, temperature=0.0, stop=["?"])

                ## cloud enable section ##
                # Generate answers
                prompt = ""
                for idx in range(len(paraphrased_df.loc[i]['choices']['label'])):
                    prompt += paraphrased_df.loc[i]['choices']['label'][idx] + ". " + paraphrased_df.loc[i]['choices']['text'][idx] + '\n'
                prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
                
                answers = generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1)

                questions_list.append(question)
                answers_list.append(answers)
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



        append2file(res_path, f"Question generated:")
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
        append2file(
            res_path,
            f"ROUGELSUM MEAN Question: {np.mean(rougeLSum_q_scores)} STD Question: {np.std(rougeLSum_q_scores)}",
        )

        append2file(res_path, f"Answer generated:")
        append2file(res_path, f"Accuracy mean: {np.mean(acc_scores)}")
        append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")


    
if __name__ == "__main__":
    run()