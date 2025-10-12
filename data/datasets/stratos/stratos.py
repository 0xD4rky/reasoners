from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset
from tqdm import tqdm

@DataFactory.register_dataset("stratos")
class StratosDataset(BaseSFTDataset):

    def load_data(self) -> Dataset:
        return load_dataset("bespokelabs/Bespoke-Stratos-17k", "main", split="train")
    
    def convert(self, example: Dict[str, Any]) -> SFTConfig:

        def replace_think_tokens(text: str) -> str:
            return text.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")

        messages = []
        for i in range(len(example)):
            messages.append({
                "role": example[i].get("from"),
                "content": replace_think_tokens(example[i].get("value"))
            })
        
        return SFTConfig(messages=messages)
    
    def parse_data(self) -> List[SFTConfig]:

        dataset = self.load_data()
        messages = []
        for i in tqdm(range(len(dataset)), desc="Parsing GSM8K dataset"):
            messages.append(self.convert(dataset[i]))
        
        return messages