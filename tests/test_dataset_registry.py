from PIL.ExifTags import Base
import pytest
from data.base import BaseSFTDataset, SFTConfig
from data.factory import DataFactory
from typing import List, Dict, Any


class MockDataset(BaseSFTDataset):
    def load_dataset(self) -> List[Dict[str, Any]]:
        message = [{
                    "from": "user",
                    "value": "Return your final response within \\boxed{}. Find [the decimal form of] the largest prime divisor of $100111011_6$.\n"
                }]

        return message

    def convert(self, example: Dict[str, Any]) -> SFTConfig:
        return SFTConfig(messages=[{"role": example[0]["from"], "content": example[0]["value"]}])


class TestDataFactory:

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        DataFactory._datasets.clear()
        yield
        DataFactory._datasets.clear()

    def test_registry_of_datasets(self):
        """
        test to verify the registry of dataset function
        """

        @DataFactory.register_dataset("test-dataset")
        class TestDataset(BaseSFTDataset):
            def load_dataset(self):
                return []
            def convert(self, example):
                return SFTConfig(messages=[])

        assert "test-dataset" in DataFactory.available_datasets()

    def test_naming_convention(self):
        """
        test to see whether each dataset is in lower case or not
        """

        @DataFactory.register_dataset("Test-Dataset")
        class TestDataset(BaseSFTDataset):
            def load_dataset(self):
                return []
            def convert(self, example):
                return SFTConfig(messages=[])

        assert "test-dataset" in DataFactory.available_datasets()
    
    def test_create_dataset_instance(self):
        """
        test to see whether the create_dataset function is able to create an instance of the dataset
        """

        @DataFactory.register_dataset("test-dataset")
        class TestDataset(BaseSFTDataset):
            def load_dataset(self):
                return []
            def convert(self, example):
                return SFTConfig(messages=[])

        dataset = DataFactory.create_dataset("test-dataset") # FIRST REGISTER, THEN CREATE
        assert isinstance(dataset, TestDataset)
    
    def test_fetching_all_dataset_info(self):
        """
        test to verify the get_all_dataset function
        """

        @DataFactory.register_dataset("test-dataset")
        class TestDataset(BaseSFTDataset):
            def load_dataset(self):
                return []
            def convert(self, example):
                return SFTConfig(messages=[])
        
        info = DataFactory.get_all_datasets()
        assert "test-dataset" in info.keys()
        assert isinstance(info["test-dataset"], type(TestDataset))
    
