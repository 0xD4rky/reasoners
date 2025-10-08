import yaml
from typing import List
from datasets import interleave_datasets, concatenate_datasets, Dataset
from data.factory import DataFactory

class DataMixer:

    def __init__(self, config_path: str):
        """
        initialize DataMixer class with configs from yaml file

        args: path to yaml file
        """

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.strategy = self.config['mixing_strategy']
        self.datasets_config = self.config['datasets']
    
    def load_datasets(self) -> Dataset:

        datasets = []

        for dataset_config in self.datasets_config:
            dataset_name = dataset_config["name"]

            dataset_instance = DataFactory.create_dataset(dataset_name)
            print(f"parsing {dataset_name} dataset \n")
            converted_data = dataset_instance.parse_data() # converting the datasets to a unified format for sft
            converted_data_dicts = [item.to_dict() for item in converted_data]
            hf_dataset = Dataset.from_list(converted_data_dicts)
            datasets.append(hf_dataset)

            print(f"Loaded {dataset_name}: {len(hf_dataset)} examples")

        return datasets
    
    def mix_datasets(self) -> Dataset:
        """
        mix datasets according to configured strategy
        """
        
        datasets = self.load_datasets()

        if self.strategy == 'interleave':
            return self._interleave_datasets(datasets)
        elif self.strategy == 'concatenate':
            return self._concatenate_datasets(datasets)
        else:
            raise ValueError(f"wrong strategy: {self.strategy}")
        
    def _interleave_datasets(self, datasets: List[Dataset]) -> Dataset:
        
        probabilities = [d['weight'] for d in self.datasets_config]

        total_weight = sum(probabilities)
        probabilities = [p / total_weight for p in probabilities]

        interleave_config = self.config.get('interleave_settings', {})
        seed = interleave_config.get('seed', 42)
        stopping_strategy = interleave_config.get('stopping_strategy', 'all_exhausted')

        mixed_dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy
        )

        print(f"Interleaved {len(datasets)} datasets with probabilities {probabilities}")
        print(f"Total examples: {len(mixed_dataset)}")

        return mixed_dataset
    
    def _concatenate_datasets(self, datasets: List[Dataset]) -> Dataset:
        
        mixed_dataset = concatenate_datasets(datasets)

        print(f"Concatenated {len(datasets)} datasets")
        print(f"Total examples: {len(mixed_dataset)}")

        return mixed_dataset
    
    def save_mixed_dataset(self, output_path: str):
        mixed_dataset = self.mix_datasets()

        import json
        with open(output_path, 'w') as f:
            json.dump(list(mixed_dataset), f, indent=2)

        print(f"final mixed dataset saved to: {output_path}")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_path", default="data/mixed_dataset.json")
    args = parser.parse_args()

    mixer = DataMixer(args.config)
    mixer.save_mixed_dataset(args.output_path)
