from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset

@DataFactory.register_dataset("gsm8k")
class GSM8KDataset(BaseSFTDataset):

    def load_data(self) -> Dataset:
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
    
    def parse_data(self) -> List[SFTConfig]:

        dataset = self.load_data()
        messages = []
        for i in range(len(dataset)):
            messages.append(self.convert(dataset[i]))
        
        return messages