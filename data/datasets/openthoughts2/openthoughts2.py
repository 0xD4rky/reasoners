from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm

@DataFactory.register_dataset("open-thoughts-2")
class OpenThoughts2Dataset(BaseSFTDataset):

    def load_data(self) -> Dataset | IterableDataset:
        return load_dataset("open-thoughts/OpenThoughts2-1M", split="train", streaming=self.streaming)
    
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
        total = getattr(dataset, "num_rows", None)
        for example in tqdm(dataset, desc="Parsing OpenThoughts2 dataset", total=total):
            messages.append(self.convert(example["conversations"]))
        
        return messages
