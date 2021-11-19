from abc import ABCMeta, abstractmethod
from typing import List, Dict


class CustomDataset(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_data_dict(self) -> Dict[str, List]:
        pass

    @abstractmethod
    def read_data_from_files(self, data_root: str, mode: str):
        pass

    @abstractmethod
    def convert_data_to_dict(self, data):
        pass
