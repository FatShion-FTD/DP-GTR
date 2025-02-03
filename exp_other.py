from utils import append2file, str_to_list
from datasets import load_dataset
import evaluate
import numpy as np
from openai_backTranslation import generate
import pandas as pd
import os, sys
from utils import convert_ABCDE, create_choices
from dpps.jointEM import joint
from dpps.DPMLM import DPMLM
from dpps.LLMDP import DPParaphrase


TEST_SIZE = 200
DATASET_TYPE = 'medQA' # medQA, csQA, VQA
EXP_TYPE = 'DPMLM' # DPMLM, DPParaphrase, HAS

temperture_list = [1,2,3,4,5,6,7,8,9,10]
# paraphrase_Ts = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# epsilon_list = []


    
def build_prompt(df, i, t):
    if DATASET_TYPE == 'csQA':
        prompt = ""
        for idx in range(len(df.loc[i]['choices']['label'])):
            prompt += df.loc[i]['choices']['label'][idx] + ". " + df.loc[i]['choices']['text'][idx] + '\n'
        prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
        return prompt
    elif DATASET_TYPE == 'medQA':
        prompt = ""
        for k,v in df.loc[i]['choices'].items():
            prompt += k + ". " + v + '\n'
        prompt = df.loc[i][f'T_{t}'] + '\nAnswer the question with the following options: ' + '\n' + prompt + 'Answer Index: '
        return prompt
    elif DATASET_TYPE == 'VQA':
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
    
def get_df():
    if DATASET_TYPE == 'csQA':
        data = load_dataset("tau/commonsense_qa")
        test_df = data['validation'].to_pandas()
        test_df = test_df.head(TEST_SIZE)
    elif DATASET_TYPE == 'medQA':
        test_df = pd.read_json(f"dataset/medQA_4.json", orient="records")
        test_df = test_df.head(TEST_SIZE)
    elif DATASET_TYPE == 'VQA':
        data = load_dataset("nielsr/docvqa_1200_examples_donut")
        test_df = data["test"].to_pandas()
        test_df = test_df.head(TEST_SIZE)
    return test_df
    
def run_DPParaphrase():
    test_df = get_df()
    paraphrase_Ts = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    logits_bias_dict = {"32": 100, "33": 100, "34": 100, "35": 100, "36": 100}     # A-E: 32-35
    path_prefix = f"results/{DATASET_TYPE}_res/{EXP_TYPE}/"
    
    for T in paraphrase_Ts:
        
        
        
        
def run_DPMLM():
    
    
    
    
def run_HAS():
    
    
    
    
if __name__ == '__main__':
    if EXP_TYPE == 'DPMLM':
        run_DPMLM()
    elif EXP_TYPE == 'DPParaphrase':
        run_DPParaphrase()
    elif EXP_TYPE == 'HAS':
        run_HAS()
    else:
        print('Invalid EXP_TYPE')
        sys.exit(0)