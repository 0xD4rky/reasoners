import json
from typing import List, Dict, Any
from base import BaseSFTDataset, SFT_Config
from factory import DataFactory
from datasets import load_dataset

@DataFactory.register_dataset("open-thoughts-2")
class OpenThoughts2Dataset(BaseSFTDataset):

    def load_dataset(self) -> List[Dict[str, Any]]:
        return load_dataset("open-thoughts/openthoughts2", split="train")
    
    def convert(self, example: Dict[str, Any]) -> SFT_Config:

        messages = []
        for i in range(len(example)):
            messages.append({
                "role": example[i]["from"],
                "content": example[i]["value"]
            })
        
        return SFT_Config(messages=messages)