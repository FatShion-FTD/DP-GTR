from utils import append2file, str_to_list
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os, sys
from utils import convert_ABCDE, create_choices
from dpps.jointEM import joint
    
    

TEST_SIZE = 200
data = load_dataset("nielsr/docvqa_1200_examples_donut")
test_df = data["test"].to_pandas()
test_df = test_df.head(TEST_SIZE)

paraphrase_Ts = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# temperture_list = [1,2,3,4,5,6,7,8,9,10]
temperture_list = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

path_prefix = "results/VQA/GPT35_DiffT/"
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)
    
REMAIN = 10
EXP_NAME = f"\n{REMAIN} Tokens Avoid Generation lowest ppl reference ICL Experiment:"
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
perplexity_metric = evaluate.load("perplexity", module_type="metric")





def build_answer_prompt(df:pd.DataFrame, i, t):
    prompt = ""
    for word in test_df.loc[i]["words"]:
        prompt += word + ", "
    prompt = (
        "Extracted OCR tokens from image:\n"
        + prompt[:-2]
        + "\nQuestion: "
        + df.loc[i][f"T_{t}"]
        + "\nAnswer the question with short term:\n"
    )
    return prompt


def run(paraphrase_T):
    res_path = f"{path_prefix}res_{paraphrase_T}.txt"
    append2file(res_path, EXP_NAME)
    append2file(res_path, f"Paraphrase T: {paraphrase_T}")

    paraphrased_df = pd.DataFrame()
    # Concat the paraphrased questions with the original questions and answers
    paraphrased_df['original_question'] = test_df['query'].apply(lambda x: x['en'])
    paraphrased_df["original_answer"] = test_df["answers"]
    paraphrased_df['words'] = test_df['words']
    
    
    # ## Step1.1: Paraphrase Questions
    T = paraphrase_T
    for tempreture in temperture_list:
        paraphrased_df[f"T_{tempreture}"] = paraphrased_df['original_question'].apply(
            lambda x: generate(
                f"Question: {x}\nParaphrase of the question :",
                temperature=tempreture,         # T for group, tempreture for diff T
                stop=["?"],
            )
        )
        
    # Check if all the paraphrased questions are empty
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
            prompt = build_answer_prompt(paraphrased_df, i, t)

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

    append2file(res_path, f"Answer paraphrased S1:")
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


    # Step2: KSA Generate Questions 
    # paraphrased_df.head()
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
            for t in temperture_list:
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
            for t in temperture_list:
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
            question = generate(icl_prompt, temperature=0.0, stop=["?"])

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
            question = generate(icl_prompt, temperature=0.0, stop=["?"])
            
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


    
if __name__ == "__main__":
    # for paraphrase_T in paraphrase_Ts:
    #     run(paraphrase_T)
    run(0)






















# from utils import append2file
# from datasets import load_dataset
# import evaluate
# import numpy as np
# from openai_backTranslation import generate
# import pandas as pd
# import os
# from dpps.LLMDP import DPParaphrase
# from dpps.DPMLM import DPMLM
# from dpps.SLM import SLM


# paraphrase_Ts = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# # paraphrase_Ts = [0.75, 1.0, 1.25, 1.5]
# # dp_Paraphrase = DPParaphrase()
# # dp_Paraphrase = DPMLM()
# # epsilon_list = [352, 176, 118, 88, 71, 59]        # DP-FT-GPT2
# # epsilon_list = [160,80,54,40,32,27]                 # DP-MLM
# # assert len(paraphrase_Ts) == len(epsilon_list)

# for T_idx, paraphrase_T in enumerate(paraphrase_Ts):
#     path_prefix = "results/VQA_res/"
#     if not os.path.exists(path_prefix):
#         os.makedirs(path_prefix)

#     res_path = f"{path_prefix}res_{paraphrase_T}.txt"
#     append2file(res_path, "\n25% Tokens Avoid lowest ppl reference ICL Experiment:")
#     # append2file(res_path, "\nDP Paraphrase(MLM) S1 Experiment:")

#     # data = load_dataset("nielsr/docvqa_1200_examples_donut")
#     # rouge_metric = evaluate.load("rouge")
#     # bleu_metric = evaluate.load("bleu")

#     # train_df = data["train"].to_pandas()
#     # test_df = data["test"].to_pandas()
#     # # test_df.head()
#     # test_df = test_df.head(200)  # TEST = 10, ALL = 200

#     # paraphrased_df = pd.DataFrame()
#     # temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#     # t = paraphrase_T
#     # for tempreture in temperture_list:
#     #     # paraphrased_df[f'T_{tempreture}'] = test_df['query'].apply(lambda x: generate(f"Document : {x['en']}\nParaphrase of the document :", tempreture, stop=["\n"]))
#     #     paraphrased_df[f"T_{tempreture}"] = test_df["query"].apply(
#     #         lambda x: generate(
#     #             f"Question : {x['en']}\nParaphrase of the question :",
#     #             temperature=t,
#     #             stop=["\n"],
#     #         )
#     #     )
    
#     # for idx, tempreture in enumerate(temperture_list):
#     #     # paraphrased_df[f"T_{tempreture}"] = test_df["query"].apply(
#     #     #     lambda x: dp_Paraphrase.privatize(
#     #     #         x["en"], epsilon=epsilon_list[T_idx]
#     #     #         )
#     #     # )
#     #     paraphrased_df[f"T_{tempreture}"] = test_df["query"].apply(
#     #         lambda x: dp_Paraphrase.dpmlm_rewrite(
#     #             x["en"], epsilon=epsilon_list[T_idx], REPLACE=True, FILTER=True, STOP=True, TEMP=True, 
#     #             POS=True, CONCAT=True)[0]
#     #     )

#     # references = test_df["query"].apply(lambda x: x["en"]).tolist()
#     # rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
#     # bleu_scores = []
#     # for tempreture in temperture_list:
#     #     predictions = paraphrased_df[f"T_{tempreture}"].tolist()
#     #     score = rouge_metric.compute(references=references, predictions=predictions)
#     #     # print(f"tempreture: {tempreture};   rouge1: {score['rouge1']}; rouge2: {score['rouge2']}; rougeL: {score['rougeL']}")
#     #     print(
#     #         f"rouge1: {score['rouge1']}; rouge2: {score['rouge2']}; rougeL: {score['rougeL']}"
#     #     )
#     #     rouge1_scores.append(score["rouge1"])
#     #     rouge2_scores.append(score["rouge2"])
#     #     rougeL_scores.append(score["rougeL"])
#     #     bleu_score = bleu_metric.compute(references=references, predictions=predictions)
#     #     bleu_scores.append(bleu_score["bleu"])
#     #     print(f"bleu: {bleu_score}")

#     # append2file(res_path, f"Question paraphrased S1:")
#     # append2file(
#     #     res_path,
#     #     f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; rougeL mean: {np.mean(rougeL_scores)}",
#     # )
#     # append2file(
#     #     res_path,
#     #     f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; rougeL std: {np.std(rougeL_scores)}",
#     # )
#     # append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
#     # append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

#     # gt_answer = []
#     # for i in range(len(test_df)):
#     #     answer = ""
#     #     for word in test_df.loc[i]["answers"]:
#     #         answer += word + ", "
#     #     answer = answer[:-2]
#     #     gt_answer.append(answer)
#     # print(gt_answer)

#     # rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []
#     # repeats = 5

#     # for t in temperture_list:
#     #     T_predictions = []
#     #     for i in range(len(test_df)):
#     #         prompt = ""
#     #         for word in test_df.loc[i]["words"]:
#     #             prompt += word + ", "
#     #         prompt = (
#     #             "Extracted OCR tokens from image:\n"
#     #             + prompt[:-2]
#     #             + "\nQuestion: "
#     #             + paraphrased_df.loc[i][f"T_{t}"]
#     #             + "\nAnswer the question with short term:\n"
#     #         )
#     #         T_predictions.append(generate(prompt, temperature=0.0, stop=["\n"]))
#     #     score = rouge_metric.compute(references=gt_answer, predictions=T_predictions)
#     #     rouge1_scores.append(score["rouge1"])
#     #     rouge2_scores.append(score["rouge2"])
#     #     rougeL_scores.append(score["rougeL"])
#     #     bleu_score = bleu_metric.compute(
#     #         references=gt_answer, predictions=T_predictions
#     #     )
#     #     bleu_scores.append(bleu_score["bleu"])

#     # append2file(res_path, f"Answer paraphrased S1:")
#     # append2file(
#     #     res_path,
#     #     f"rouge1 mean: {np.mean(rouge1_scores)}; rouge2 mean: {np.mean(rouge2_scores)}; rougeL mean: {np.mean(rougeL_scores)}",
#     # )
#     # append2file(
#     #     res_path,
#     #     f"rouge1 std: {np.std(rouge1_scores)}; rouge2 std: {np.std(rouge2_scores)}; rougeL std: {np.std(rougeL_scores)}",
#     # )
#     # append2file(res_path, f"bleu mean: {np.mean(bleu_scores)}")
#     # append2file(res_path, f"bleu std: {np.std(bleu_scores)}")

#     # # Concat the paraphrased questions with the original questions and answers
#     # paraphrased_df["original_question"] = test_df["query"].apply(lambda x: x["en"])
#     # paraphrased_df["original_answer"] = test_df["answers"]
#     # paraphrased_df["words"] = test_df["words"]

#     df_file_path = f"{path_prefix}GPT35_JSON/paraphrased_questions_T{paraphrase_T}.json"

#     # # # Save the paraphrased questions
#     # paraphrased_df.to_json(df_file_path, orient="records")


#     # # Step2: KSA Generate Questions 
#     paraphrased_df = pd.read_json(df_file_path)
#     # paraphrased_df.head()
#     questions_gt_list = paraphrased_df["original_question"].tolist()
#     answers_gt_list = paraphrased_df["original_answer"].tolist()
#     paraphrased_df = paraphrased_df.fillna(" ")

#     rouge_metric = evaluate.load("rouge")
#     bleu_metric = evaluate.load("bleu")
#     perplexity_metric = evaluate.load("perplexity", module_type="metric")
    
#     import nltk
#     nltk.download("stopwords")
#     nltk.download("punkt")
#     from nltk.corpus import stopwords
#     from nltk.util import ngrams

#     stopword_set = set(stopwords.words("english"))
#     import string
#     import random

#     temperture_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#     repeats = 5

#     (
#         bleu_q_scores,
#         bleu_a_scores,
#         rouge1_q_scores,
#         rouge2_q_scores,
#         rougeL_q_scores,
#         rouge1_a_scores,
#         rouge2_a_scores,
#         rougeL_a_scores,
#     ) = ([], [], [], [], [], [], [], [])
    
#     analyt_df = pd.DataFrame(columns=["Reference Question", "Generated Question"])
#     for idx, t in enumerate(temperture_list):
#         analyt_df[f"P_{idx}"] = paraphrased_df[f"T_{t}"]
#     analyt_df_save_path = f"{path_prefix}GPT35_JSON/analyt_df_T{paraphrase_T}.json"

#     for r in range(repeats):
#         questions_list = []
#         answers_list = []
#         reference_question_list = []
#         generated_question_list = []
        
#         for i in range(len(paraphrased_df)):
#             # Count token freq
#             all_tokens = {}  # key: token, value: count
#             for t in temperture_list:
#                 sentence = paraphrased_df.loc[i][f"T_{t}"]
#                 tokens = nltk.word_tokenize(sentence)
#                 onegrams = set(ngrams(tokens, 1))
#                 # onegrams = set(onegrams)
#                 # making onegrams a set to avoid duplicate tokens
#                 for token in onegrams:
#                     # only add one gram for one sentence
#                     if token in all_tokens:
#                         all_tokens[token] += 1
#                     else:
#                         all_tokens[token] = 1
#             print(f"All Tokens:  {all_tokens}")

#             # ================ Add Noise Here ================
#             all_tokens_sorted = sorted(
#                 all_tokens.items(), key=lambda x: x[1], reverse=True
#             )
#             print(f"All Sorted Tokens:  {all_tokens_sorted}")
#             # ignore those non-words tokens
#             filtered_tokens = {}
#             for token, count in all_tokens_sorted:
#                 if (
#                     not all(word in string.punctuation for word in token)
#                     and token[0] not in stopword_set
#                 ):
#                     filtered_tokens[token] = count
#             filtered_tokens_sorted = sorted(
#                 filtered_tokens.items(), key=lambda x: x[1], reverse=True
#             )
#             print(f"Filtered Sorted Tokens:  {filtered_tokens_sorted}")

#             # Non-DP: find the count threshold where the count gap is the largest
#             # actually_upper_bound = 0
#             # count_threshold = (
#             #     0  # you need to creat a list to store all difference between the counts
#             # )
#             # for k in range(len(filtered_tokens_sorted) - 1):
#             #     # print(k,len(filtered_tokens_sorted)-2)
#             #     if (
#             #         filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k + 1][1]
#             #         > count_threshold
#             #     ):
#             #         count_threshold = (
#             #             filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k + 1][1]
#             #         )
#             #         actually_upper_bound = filtered_tokens_sorted[k][1]
#             #     if k == len(filtered_tokens_sorted) - 2:
#             #         if filtered_tokens_sorted[k + 1][1] > count_threshold:
#             #             count_threshold = filtered_tokens_sorted[k + 1][1]
#             #             actually_upper_bound = filtered_tokens_sorted[k + 1][
#             #                 1
#             #             ]  # including all tokens

#             # # print(f"count_threshold: {count_threshold}")
#             # filtered_tokens = dict(filtered_tokens_sorted)
#             # filtered_tokens = [
#             #     k[0] for k, v in filtered_tokens.items() if v >= actually_upper_bound
#             # ]

#             # Non-DP: Top 5 tokens
#             # if len(filtered_tokens_sorted) > 5:
#             #     filtered_tokens_sorted = filtered_tokens_sorted[:5]
#             # filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
#             # print(filtered_tokens)
            
#             # Non-DP: 75% tokens, 50% tokens, 25% tokens
#             filtered_tokens_sorted = filtered_tokens_sorted[: int(len(filtered_tokens_sorted) * 0.25)]
#             filtered_tokens = [k[0][0] for k in filtered_tokens_sorted]
#             print(filtered_tokens)

#             # DP: RNM-Gaussian

#             # ================ End Here ================

#             random.shuffle(filtered_tokens)  # shuffle the list of tokens

#             # Build tokens prompt
#             suggest_tokens = ""
#             for token in filtered_tokens:
#                 suggest_tokens += token + ", "
#             suggest_tokens = suggest_tokens[:-2]
            
#             # lowest ppl reference question
#             paraphrase_sentences = []
#             for t in temperture_list:
#                 if len(paraphrased_df.loc[i][f"T_{t}"]) > 0:
#                     paraphrase_sentences.append(paraphrased_df.loc[i][f"T_{t}"])
#                 else:
#                     paraphrase_sentences.append(" ")
                    
#             perplexity_res = perplexity_metric.compute(predictions=paraphrase_sentences, model_id="gpt2")
#             tmp_df = pd.DataFrame({"Predictions": paraphrase_sentences, "Perplexity": perplexity_res['perplexities']})
#             lowest_perplexity_idx = tmp_df["Perplexity"].idxmin()
#             reference_question = tmp_df.loc[lowest_perplexity_idx]["Predictions"]

#             # Build Prompt and generate questions
#             icl_prompt = (
#                     "Paraphrase the following question:\n"
#                     + reference_question
#                     + "\nAvoid using following tokens:\n"
#                     # + "\nDo not using following tokens:\n"
#                     + suggest_tokens
#                     + "\nParaphrased question:"
#                 )
#             question = generate(icl_prompt, temperature=0.0, stop=["?"])
            
#             reference_question_list.append(reference_question)
#             generated_question_list.append(question)

#             ## cloud enable section ##
#             # Generate answers
#             prompt = ""
#             for word in paraphrased_df.loc[i]["words"]:
#                 prompt += word + ", "
#             prompt = (
#                 "Extracted OCR tokens from image:\n"
#                 + prompt[:-2]
#                 + "\nQuestion: "
#                 + question
#                 + "\nAnswer the question with short term:\n"
#             )
#             answers = generate(prompt, temperature=0.0, stop=["\n"])

#             questions_list.append(question)
#             answers_list.append(answers)
        
#         analyt_df[f"R{r}_RQ"] = reference_question_list
#         analyt_df[f"R{r}_GQ"] = generated_question_list
        
#         bleu_q_score = bleu_metric.compute(
#             references=questions_gt_list, predictions=questions_list
#         )
#         rouge_q_score = rouge_metric.compute(
#             references=questions_gt_list, predictions=questions_list
#         )
#         bleu_ans_score = bleu_metric.compute(
#             references=answers_gt_list, predictions=answers_list
#         )
#         rouge_ans_score = rouge_metric.compute(
#             references=answers_gt_list, predictions=answers_list
#         )
#         bleu_q_scores.append(bleu_q_score["bleu"])
#         rouge1_q_scores.append(rouge_q_score["rouge1"])
#         rouge2_q_scores.append(rouge_q_score["rouge2"])
#         rougeL_q_scores.append(rouge_q_score["rougeL"])
#         bleu_a_scores.append(bleu_ans_score["bleu"])
#         rouge1_a_scores.append(rouge_ans_score["rouge1"])
#         rouge2_a_scores.append(rouge_ans_score["rouge2"])
#         rougeL_a_scores.append(rouge_ans_score["rougeL"])


#     analyt_df.to_json(analyt_df_save_path, orient="records")
    
#     append2file(res_path, f"Question generated:")
#     append2file(
#         res_path,
#         f"BLEU MEAN Question: {np.mean(bleu_q_scores)} STD Question: {np.std(bleu_q_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGE1 MEAN Question: {np.mean(rouge1_q_scores)} STD Question: {np.std(rouge1_q_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGE2 MEAN Question: {np.mean(rouge2_q_scores)} STD Question: {np.std(rouge2_q_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGEL MEAN Question: {np.mean(rougeL_q_scores)} STD Question: {np.std(rougeL_q_scores)}",
#     )

#     append2file(res_path, f"Answer generated:")
#     append2file(
#         res_path,
#         f"BLEU MEAN Answer: {np.mean(bleu_a_scores)} STD Answer: {np.std(bleu_a_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGE1 MEAN Answer: {np.mean(rouge1_a_scores)} STD Answer: {np.std(rouge1_a_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGE2 MEAN Answer: {np.mean(rouge2_a_scores)} STD Answer: {np.std(rouge2_a_scores)}",
#     )
#     append2file(
#         res_path,
#         f"ROUGEL MEAN Answer: {np.mean(rougeL_a_scores)} STD Answer: {np.std(rougeL_a_scores)}",
#     )
