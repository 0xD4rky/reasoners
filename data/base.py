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

    """
    Base class for all SFT datasets to convert into a unified format {"role": "user", "content": "..."} for sft
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_data(self) -> Dataset:
        pass
    
    @abstractmethod
    def convert(self, example: Dict[str, Any]) -> SFTConfig:
        pass
    
    @abstractmethod
    def parse_data(self) -> List[SFTConfig]:
        pass
    
    def save_data(self, output_path: str):

        dataset = self.parse_dataset()
        with open(output_path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example.to_dict()) + "\n")

        print(f"Dataset saved to {output_path}, length: {len(dataset)}")