from data.base import BaseSFTDataset
from typing import Dict, List, Type


class DataFactory:

    _datasets: Dict[str, Type[BaseSFTDataset]] = {}

    @classmethod
    def register_dataset(cls, name: str):
        """
        class decorator to register a dataset handler under `name`

        eg:
            @DatasetFactory.register_dataset("open-thoughts")
            class OpenThoughtsDataset(BaseSFTDataset):
                pass
        """
        def decorator(dataset_class: Type[BaseSFTDataset]):
            key = name.strip().lower()
            if key in cls._datasets:
                raise ValueError(f"Dataset '{key}' is already registered.")
            cls._datasets[key] = dataset_class
            return dataset_class
        
        return decorator
    
    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> BaseSFTDataset:
        """
        function for creating an instance of a registered dataset
        """
        key = name.strip().lower()
        if key not in cls._datasets:
            raise ValueError(f"Unknown dataset: '{name}'. Available: {list(cls._datasets.keys())}")
        dataset_class = cls._datasets[key]
        return dataset_class(**kwargs)
    
    @classmethod
    def available_datasets(cls) -> List[str]:
        return (list(cls._datasets.keys()))
    
    @classmethod
    def get_all_datasets(cls) -> Dict[str, Type[BaseSFTDataset]]:
        return cls._datasets.copy()
