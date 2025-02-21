from datasets import load_dataset
import pandas as pd

class CustomDataset:
    """
    Custom dataset loader for various dataset types.
    
    Supported types:
      - 'csqa': Loads CommonsenseQA validation data.
      - 'medqa': Loads local medQA data (expects 'dataset/medQA_4.json').
      - 'vqa':  Loads VQA data from the docvqa dataset.
    
    Parameters:
      dataset_type (str): Type of the dataset to load ('csqa', 'medqa', or 'vqa').
      test_size (int): Number of examples to load (default: 200).
    """
    def __init__(self, dataset_type: str, test_size: int = 200):
        self.dataset_type = dataset_type.lower()
        self.test_size = test_size
        self.data = pd.DataFrame()
        tmp_data = self._load()
        if self.dataset_type == "csqa":
            self.data['original_question'] = tmp_data['question']
            self.data['original_answer'] = tmp_data['answerKey']
            self.data['choices'] = tmp_data['choices']
        elif self.dataset_type == "medqa":
            self.data['original_question'] = tmp_data['question']
            self.data['original_answer'] = tmp_data['answer_idx']
            self.data['choices'] = tmp_data['options']
        elif self.dataset_type == "vqa":
            self.data['original_question'] = tmp_data['query'].apply(lambda x: x['en'])
            self.data['original_answer'] = tmp_data['answers']
            self.data['words'] = tmp_data['words']
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            

    def _load(self) -> pd.DataFrame:
        """
        Load the specified dataset and return it as a pandas DataFrame.
        
        Returns:
          pd.DataFrame: The loaded dataset with test_size examples.
        """
        if self.dataset_type == "csqa":
            data = load_dataset("tau/commonsense_qa")
            return data["validation"].to_pandas().head(self.test_size)
        elif self.dataset_type == "medqa":
            data = load_dataset("GBaker/MedQA-USMLE-4-options")
            return data["test"].to_pandas().head(self.test_size)
        elif self.dataset_type == "vqa":
            data = load_dataset("nielsr/docvqa_1200_examples_donut")
            return data["test"].to_pandas().head(self.test_size)
