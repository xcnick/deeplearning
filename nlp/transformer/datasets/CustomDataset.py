from abc import ABCMeta, abstractmethod
from typing import List, Any


class CustomDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def read_examples_from_files(self, data_root: str, mode: str) -> List[Any]:
        pass

    @abstractmethod
    def get_dataset(self, examples: List[Any]):
        pass
