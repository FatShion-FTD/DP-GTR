import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList
)

class ClipLogitsProcessor(LogitsProcessor):
  def __init__(self, min=-100, max=100):
    self.min = min
    self.max = max

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    scores = torch.clamp(scores, min=self.min, max=self.max)
    return scores


class StopOnToken(StoppingCriteria):
    """Custom stopping criteria to stop generation when a specific token is encountered."""
    def __init__(self, tokenizer, stop_words=None):
        if stop_words is None:
            stop_words = ["?", "\n", "?\n", "? ", " ?"]
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0, -1].item()
        last_token = self.tokenizer.decode([last_token_id], skip_special_tokens=False)
        # print(f"Last Token Decoded: {last_token}")      # DEBUG
        return last_token in self.stop_words

class SLM:
    """A class to handle small language models (SLM) for text generation tasks."""

    MODELS = {
        "TinyLlama": {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "all_clip": {
                "mean": 15.362243,
                "std": 4.701108
            },
            "single_clip": {
                "mean": 16.199137,
                "std": 4.245684
            }
        },
        "Llama3.2-3B": {
            "model_name": "meta-llama/Llama-3.2-3B",
            "all_clip": {
                "mean": 19.003282,
                "std": 3.511572
            },
            "single_clip": {
                "mean": 20.064617,
                "std": 3.135530
            }
        },
        "gpt-neo-1.3B": {
            "model_name": "EleutherAI/gpt-neo-1.3B",
            "all_clip": {
                "mean": -5.133257,
                "std": 2.198805
            },
            "single_clip": {
                "mean": -0.998723,
                "std": 2.123457
            }
        },
        "OPT-1.3B": {
            "model_name": "facebook/opt-1.3b",
            "all_clip": {
                "mean": 8.834933,
                "std": 1.584638
            },
            "single_clip": {
                "mean": 13.012060,
                "std": 3.538185
            }
        },
        "pythia-1.4b": {
            "model_name": "EleutherAI/pythia-1.4b",
            "all_clip": {
                "mean": 11.317303,
                "std": 1.320636
            },
            "single_clip": {
                "mean": 15.436553,
                "std": 3.141557
            }
        },
        "LiteLlama-460M": {
            "model_name": "ahxt/LiteLlama-460M-1T",
            "all_clip": {
                "mean": 4.568743,
                "std": 12.465012
            },
            "single_clip": {
                "mean": 8.488522,
                "std": 13.585919
            }
        },
        "Qwen2-1.5B": {
            "model_name": "Qwen/Qwen2-1.5B",
            "all_clip": {
                "mean": 13.940535,
                "std": 1.822326
            },
            "single_clip": {
                "mean": 20.427131,
                "std": 3.847943
            }
        },
        "SmolLM-1.7B": {
            "model_name": "HuggingFaceTB/SmolLM-1.7B",
            "all_clip": {
                "mean": 18.290090,
                "std": 3.484005
            },
            "single_clip": {
                "mean": 22.968606,
                "std": 4.377783
            }
        },
        "Llama-3.2-1B": {
            "model_name": "meta-llama/Llama-3.2-1B",
            "all_clip": {
                "mean": 21.433960,
                "std": 3.576826
            },
            "single_clip": {
                "mean": 22.531474,
                "std": 3.230357
            }
        }, 
        "Llama-3.1-8B":{
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "all_clip": {
                "mean": 22.944258,
                "std": 2.420791,
                "max": 31.031375
            },
            "single_clip": {
                "mean": 31.031375,
                "std": 6.170953
            }
        }
    }
    
    tokenizer = None
    model = None
    stopping_criteria = None
    generation_config = None
    min_logit = None
    max_logit = None
    sensitivity = None
    logits_processor = None
    model_name = None

    def __init__(self, model_name, stop_words=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code="OpenELM" in model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code="OpenELM" in model_name
        )
        self.model_name = model_name
        self.stopping_criteria = StoppingCriteriaList(
            [StopOnToken(self.tokenizer, stop_words)] if stop_words else [StopOnToken(self.tokenizer)]
        )
        
        self.generation_config = self._default_config()
        if "meta-llama" in model_name or "EleutherAI" in model_name or "Qwen" in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.generation_config.update({
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            })

    def _default_config(self):
        """Returns the default configuration for text generation."""
        return {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        
    def clip_model(self, epsilon=100, clip_type="all_clip"):
        """
        Configure the model's logits clipping based on specified epsilon and clip type.
        Parameters:
            epsilon (float): Sensitivity parameter to adjust the temperature.
            clip_type (str): Either "all_clip" or "single_clip" to specify clipping configuration.
        """
        if clip_type not in ["all_clip", "single_clip"]:
            raise ValueError("clip_type must be either 'all_clip' or 'single_clip'")

        # Retrieve mean and std for the specified clip_type
        model_config = next(
            (v for k, v in self.MODELS.items() if v["model_name"] == self.model_name),
            None
        )

        if not model_config:
            raise ValueError(f"Model {self.model_name} not found in SLM.MODELS")

        mean = model_config[clip_type]["mean"]
        std = model_config[clip_type]["std"]

        # Calculate min and max logits for clipping
        self.max_logit = mean + 4 * std
        self.min_logit = mean
        

        # Calculate sensitivity and set temperature
        self.sensitivity = abs(self.max_logit - self.min_logit)
        temperature = 2 * self.sensitivity / epsilon

        # Update logits processor
        self.logits_processor = LogitsProcessorList([ClipLogitsProcessor(self.min_logit, self.max_logit)])
        self.set_config(temperature=temperature, logits_processor=self.logits_processor)
        
    def unclip_model(self):
        """Reset the model's logits clipping configuration."""
        self.set_config(logits_processor=None, temperature=1e-3)
        

    def set_config(self, **kwargs):
        """Update generation configuration with custom parameters."""
        self.generation_config.update(kwargs)

    def get_config(self):
        """Retrieve the current generation configuration."""
        return self.generation_config

    def generate(self, input_text, ref_text=None):
        """
        Generate text based on the input_text and optional pure_text to adjust max_new_tokens.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        if ref_text and ref_text != input_text:
            ref_text_ids = self.tokenizer(ref_text, return_tensors="pt").input_ids
            input_length = ref_text_ids.shape[-1]
            self.set_config(max_new_tokens=input_length + 20)
        elif ref_text:
            ref_text_ids = inputs.input_ids
            input_length = ref_text_ids.shape[-1]
            self.set_config(max_new_tokens=input_length + 20)

        output_ids = self.model.generate(
            **inputs, stopping_criteria=self.stopping_criteria, **self.generation_config
        )
        output_text = self.tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
        # output_text = output_text.replace(input_text, "").replace(input_text.strip(), "")
        return {"output_text": output_text, "output_ids": output_ids}
    
    def clean_text(self, text, input_text):
        """Clean the generated text by removing the input_text."""
        return text.replace(input_text, "").replace(input_text.strip(), "")
    
    def generate_clean(self, input_text, ref_text=None):
        output_text = self.generate(input_text, ref_text)['output_text']
        clean_text = self.clean_text(output_text, input_text)
        return clean_text

    def check_logits(self, output_ids):
        """Check and return the mean logits for each step of the generation."""
        all_logits = []
        for logits in output_ids.scores:
            token_logits = logits[0]  # batch size = 1
            valid_logits = token_logits[token_logits != float("-inf")]
            all_logits.append(valid_logits.mean().item())
        return all_logits

    def check_token_logits(self, output_ids):
        """Check and return the logits of the final selected tokens during generation."""
        all_logits = []
        for i, logits in enumerate(output_ids.scores):
            token_logits = logits[0]  # batch size = 1
            selected_token_id = output_ids.sequences[0, len(output_ids.sequences[0]) - len(output_ids.scores) + i]
            selected_token_logit = token_logits[selected_token_id].item()
            all_logits.append(selected_token_logit)
        return all_logits
    
    def get_tokenid(self, token_list:list):
        """Get the token id for the given token list."""
        token_ids = []
        for token in token_list:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            token_ids.append(tid)
        return token_ids
        

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct" #"meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-1B"
    model = SLM(model_name)
    tokens_list = ['A', 'B', 'C', 'D', 'E']     # [32, 33, 34, 35, 36]

    model.unclip_model()
    print(f"Current Config: {model.get_config()}")
    text = "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
    paraphrased_text = model.generate(f"Paraphrase the following question:\n{text}\nParaphrased Question:\n", text)['output_text']
    paraphrased_text = model.clean_text(paraphrased_text, text)
    print(f"Paraphrased Text: {paraphrased_text}")

