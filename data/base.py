from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import json

from datasets import Dataset

@dataclass
class SFTConfig:
    messages: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]: # returns a dict of the dataclass
        return asdict(self)


class BaseSFTDataset(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_dataset(self) -> Dataset:
        pass
    
    @abstractmethod
    def convert(self, example: Dict[str, Any]) -> SFTConfig:
        pass
    
    def parse_dataset(self) -> List[SFTConfig]:
        """
        function to parse and convert the dataset into a unified format for SFT
        """

        dataset = self.load_dataset()
        return [self.convert(example) for example in dataset]
    
    def save_dataset(self, output_path: str):

        dataset = self.parse_dataset()
        with open(output_path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example.to_dict()) + "\n")

        print(f"Dataset saved to {output_path}, length: {len(dataset)}")