from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset

@DataFactory.register_dataset("gsm8k")
class GSM8KDataset(BaseSFTDataset):

    def load_dataset(self) -> Dataset:
        return load_dataset("gsm8k", "main", split="train")
    
    def convert(self, example: Dict[str, Any]) -> SFTConfig:

        messages = [{
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
        }]
        
        return SFTConfig(messages=messages)