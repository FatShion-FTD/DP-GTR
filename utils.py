import pandas as pd
import numpy as np
import ast
import re

def prompt_template(doc):
    prompt = "Document: {}\nParaphrase of the document:".format(doc)
    return prompt


def convert_ABCDE(x : list):
    new_x = []
    for i in x:
        if i == 'A':
            new_x.append(0)
        elif i == 'B':
            new_x.append(1)
        elif i == 'C':
            new_x.append(2)
        elif i == 'D':
            new_x.append(3)
        elif i == 'E':
            new_x.append(4)
        else:
            new_x.append(2)
    return new_x


def create_choices(df, n_choices):
    new_df = df.copy()
    rows = []
    for _, row in df.iterrows():
        choices = row['choices']
        answer_label = row['original_answer']
        answer_index = list(choices['label']).index(answer_label)
        answer_text = choices['text'][answer_index]
        
        other_choices = [(label, text) for label, text in zip(choices['label'], choices['text']) if label != answer_label]
        selected_other_choices = np.random.choice(len(other_choices), n_choices - 1, replace=False)
        selected_choices = [other_choices[i] for i in selected_other_choices]
        
        final_choices = [(answer_label, answer_text)] + selected_choices
        np.random.shuffle(final_choices)  
        
        new_choices = {
            'label': np.array([c[0] for c in final_choices]),
            'text': np.array([c[1] for c in final_choices])
        }
        
        new_row = {}
        for key in row.keys():
            if key != 'choices':
                new_row[key] = row[key]
        new_row['choices'] = new_choices
        
        rows.append(new_row)
    return pd.DataFrame(rows)


def str_to_list(cell):
    if pd.isna(cell) or not isinstance(cell, str):
        return []
    cell = cell.replace("\n", " ")
    cell = re.sub(r"'\s+'", "', '", cell)
    try:
        parsed_list = ast.literal_eval(cell)
        if isinstance(parsed_list, list):
            return [item.strip("'") for item in parsed_list]
    except (ValueError, SyntaxError):
        pass
    return []


def append2file(file_path, content):
    with open(file_path, 'a') as f:
        f.write(content)
        f.write('\n')