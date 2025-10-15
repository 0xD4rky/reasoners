from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm

@DataFactory.register_dataset("gsm8k")
class GSM8KDataset(BaseSFTDataset):

    def load_data(self) -> Dataset | IterableDataset:
        return load_dataset("gsm8k", "main", split="train", streaming=self.streaming)
    
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
        total = getattr(dataset, "num_rows", None)
        for example in tqdm(dataset, desc="Parsing GSM8K dataset", total=total):
            messages.append(self.convert(example))
        
        return messages
