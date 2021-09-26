from abc import ABCMeta, abstractmethod
from typing import List, Any, Dict


class CustomDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_data_dict(self) -> Dict[str, List]:
        pass

    @abstractmethod
    def read_examples_from_files(self, data_root: str, mode: str) -> List[Any]:
        pass

    @abstractmethod
    def convert_examples_to_data_dict(self, examples: List[Any]):
        pass
