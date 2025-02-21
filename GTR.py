import nltk
import evaluate
import string
import numpy as np
import pandas as pd
import random

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from nltk.util import ngrams
from dpps.jointEM import joint


class GTR:
    def __init__(self, num_rewrites=10, releasing_strategy="ndp", remian_tokens=10):
        self.perplexity_metric = evaluate.load("perplexity", module_type="metric")
        self.releasing_strategy = releasing_strategy
        self.remian_tokens = remian_tokens
        self.num_rewrites = num_rewrites
        self.stopword_set = set(stopwords.words("english"))


    def gtr(self, text, rewrite_function=None, **kwargs):
        if rewrite_function is None:
            raise ValueError("No rewrite function provided.")
        rewrites = []
        for _ in range(self.num_rewrites):
            rewritten = rewrite_function(text, **kwargs)
            rewrites.append(rewritten)
        return rewrites

    def icl(self, rewrites:list, rewrite_function=None, **kwargs):
        # ========================== Privacy ==========================
        # Count token freq
        all_tokens = {}  # key: token, value: count
        for rewrite in rewrites:
            tokens = nltk.word_tokenize(rewrite)
            onegrams = set(ngrams(tokens, 1))
            for token in onegrams:
                # only add one gram per sentence
                if token in all_tokens:
                    all_tokens[token] += 1
                else:
                    all_tokens[token] = 1
        print(f"All Tokens:  {all_tokens}")
        
        all_tokens_sorted = sorted(
            all_tokens.items(), key=lambda x: x[1], reverse=True
        )
        print(f"All Sorted Tokens:  {all_tokens_sorted}")
        # ignore those non-words tokens
        filtered_tokens = {}
        for token, count in all_tokens_sorted:
            if (
                not all(word in string.punctuation for word in token)
                and token[0] not in self.stopword_set
            ):
                filtered_tokens[token] = count
        filtered_tokens_sorted = sorted(
            filtered_tokens.items(), key=lambda x: x[1], reverse=True
        )
        print(f"Filtered Sorted Tokens:  {filtered_tokens_sorted}")
        
        if self.releasing_strategy == "ndp":
            filtered_tokens_sorted_ndp = filtered_tokens_sorted[:min(self.remian_tokens, len(filtered_tokens_sorted))]
            released_tokens = [k[0][0] for k in filtered_tokens_sorted_ndp]
            print(released_tokens)
        elif self.releasing_strategy == "jem":
            item_counts = np.array([count for token, count in filtered_tokens_sorted])
            joint_out = joint(item_counts, k=min(self.remian_tokens, len(item_counts)), epsilon=2, neighbor_type=1)
            filtered_tokens_sorted_jem = np.array(filtered_tokens_sorted, dtype=object)[joint_out]
            released_tokens = [token_tuple[0][0] for token_tuple in filtered_tokens_sorted_jem]
            print(released_tokens)
        else:
            raise ValueError(f"Unsupported releasing strategy: {self.releasing_strategy}")
        random.shuffle(released_tokens)
        
        # ========================== Utility ==========================
        paraphrase_sentences = []
        for rewrite in rewrites:
            if len(rewrite) > 0:
                paraphrase_sentences.append(rewrite)
            else:
                paraphrase_sentences.append(" ")
        perplexity_res = self.perplexity_metric.compute(predictions=paraphrase_sentences, model_id="gpt2")
        tmp_df = pd.DataFrame({"Predictions": paraphrase_sentences, "Perplexity": perplexity_res['perplexities']})
        lowest_perplexity_idx = tmp_df["Perplexity"].idxmin()
        reference_question = tmp_df.loc[lowest_perplexity_idx]["Predictions"]
        
        # ========================== Fin Prompt ==========================
        suggest_tokens = ""
        for token in released_tokens:
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
        
        fin_prompt = rewrite_function(icl_prompt, **kwargs)
        return fin_prompt
    