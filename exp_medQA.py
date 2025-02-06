from utils import append2file, str_to_list
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os, sys
from utils import convert_ABCDE, create_choices
from dpps.jointEM import joint
    
    
    

# dataset_path = f"dataset/commonsense_qa_{N}_choices.json"
# if N == 5:
#     data = load_dataset("tau/commonsense_qa")
#     test_df = data['validation'].to_pandas()
# else:
#     test_df = pd.read_json(dataset_path, orient="records", lines=True)

TEST_SIZE = 200
N = 4

data = pd.read_json(f"dataset/medQA_{N}.json", orient="records")
test_df = data
test_df = test_df.head(TEST_SIZE)

# paraphrase_Ts = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5]
paraphrase_Ts = [1.25]
logits_bias_dict = {"32": 100, "33": 100, "34": 100, "35": 100, "36": 100}     # A-E: 32-35
temperture_list = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] # [1,2,3,4,5,6,7,8,9,10]

path_prefix = "results/medQA/GPT35_DiffT/"
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)
    
REMAIN = 10
EXP_NAME = f"\n{REMAIN} Tokens Avoid Generation lowest ppl reference ICL Experiment:"
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
acc_metric = evaluate.load("accuracy")
perplexity_metric = evaluate.load("perplexity", module_type="metric")





def build_answer_prompt(df, i, t):
    prompt = ""
    for k,v in df.loc[i]['choices'].items():
        prompt += k + ". " + v + '\n'
    prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
    return prompt


def run(paraphrase_T):
    res_path = f"{path_prefix}res_{paraphrase_T}.txt"
    append2file(res_path, EXP_NAME)
    append2file(res_path, f"Paraphrase T: {paraphrase_T}")

    paraphrased_df = pd.DataFrame()
    # Concat the paraphrased questions with the original questions and answers
    paraphrased_df['original_question'] = test_df['question']
    paraphrased_df["original_answer"] = test_df["answer_idx"]
    paraphrased_df['choices'] = test_df['options']
    
    
    # ## Step1.1: Paraphrase Questions
    T = paraphrase_T
    for tempreture in temperture_list:
        paraphrased_df[f"T_{tempreture}"] = paraphrased_df['original_question'].apply(
            lambda x: generate(
                f"Question: {x}\nParaphrase of the question :",
                temperature=T,
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
    gt_answer = paraphrased_df["original_answer"].tolist()
    gt_answer = convert_ABCDE(gt_answer)
    
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
    append2file(res_path, f"Accuracy std: {np.std(acc_scores)}")
    append2file(res_path, ">" * 50)


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
            for k,v in paraphrased_df.loc[i]['choices'].items():
                prompt += k + ". " + v + '\n'
            prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            
            answers = generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1)

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
            for k,v in paraphrased_df.loc[i]['choices'].items():
                prompt += k + ". " + v + '\n'
            prompt = question + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
            
            answers = generate(prompt, temperature=0.0, logits_dict=logits_bias_dict, max_tokens=1)
            
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


    
if __name__ == "__main__":
    # for paraphrase_T in paraphrase_Ts:
    #     run(paraphrase_T)
    run(0)