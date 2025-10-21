from typing import List, Dict, Any
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm

@DataFactory.register_dataset("stratos")
class StratosDataset(BaseSFTDataset):

    def load_data(self) -> Dataset | IterableDataset:
        return load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", streaming=self.streaming)
    
    def convert(self, example: Dict[str, Any]) -> SFTConfig:

        def replace_think_tokens(text: str) -> str:
            return text.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")

        messages = [
            {
                "role": example[0].get("from"),
                "content": example[0].get("value")
            },
            {
                "role": example[1].get("from"),
                "content": replace_think_tokens(example[1].get("value"))
            }
        ]
        
        return SFTConfig(messages=messages)
    
    def parse_data(self) -> List[SFTConfig]:

        dataset = self.load_data()
        messages = []
        total = getattr(dataset, "num_rows", None)
        for example in tqdm(dataset, desc="Parsing stratos dataset", total=total):
            messages.append(self.convert(example["conversations"]))
        
        return messages
