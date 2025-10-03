import json
from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset

@DataFactory.register_dataset("open-thoughts-2")
class OpenThoughts2Dataset(BaseSFTDataset):

    def load_dataset(self) -> Dataset:
        return load_dataset("open-thoughts/openthoughts2", split="train")
    
    def convert(self, example: Dict[str, Any]) -> SFTConfig:

        messages = []
        for i in range(len(example)):
            messages.append({
                "role": example[i]["from"],
                "content": example[i]["value"]
            })
        
        return SFTConfig(messages=messages)