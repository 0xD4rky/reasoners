from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset
from tqdm import tqdm

@DataFactory.register_dataset("open-thoughts-2")
class OpenThoughts2Dataset(BaseSFTDataset):

    def load_data(self) -> Dataset:
        return load_dataset("open-thoughts/OpenThoughts2-1M", split="train")
    
    def convert(self, example: List[Dict[str, Any]]) -> SFTConfig:

        messages = []
        for i in range(len(example)):
            messages.append({
                "role": example[i].get("from"),
                "content": example[i].get("value")
            })
        
        return SFTConfig(messages=messages)
    
    def parse_data(self) -> List[SFTConfig]:

        dataset = self.load_data()
        messages = []
        for i in tqdm(range(len(dataset)), desc="Parsing OpenThoughts2 dataset"):
            messages.append(self.convert(dataset[i]["conversations"]))
        
        return messages
