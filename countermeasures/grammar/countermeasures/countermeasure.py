from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

T = TypeVar('T', bound='Countermeasure')
tqdm.pandas()


class Countermeasure(ABC):
    def __init__(self):
        pass

    def apply_on_traces(self, traces: pd.DataFrame) -> pd.DataFrame:
        return traces.progress_apply(lambda x: self.apply_on_trace(x.to_numpy()), axis=1, result_type='broadcast')

    @abstractmethod
    def apply_on_trace(self, trace: np.ndarray):
        pass

    @abstractmethod
    def get_cost(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __ne__(self, other):
        return not self == other

    @abstractmethod
    def __eq__(self, other):
        pass

    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))

    @staticmethod
    @abstractmethod
    def countermeasure_id():
        pass

    @classmethod
    @abstractmethod
    def get_instances_within_budget(cls, budget: float, noise_scale: float) -> List[T]:
        pass
