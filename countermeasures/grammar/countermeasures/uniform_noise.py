import random
from typing import List

import numpy as np

from . import countermeasure


class UniformNoise(countermeasure.Countermeasure):
    def __init__(self, noise_factor, noise_scale):
        # def __init__(self, noise_factor, noise_scale=12.5):  # For ASCAD
        # def __init__(self, noise_factor, noise_scale=0.01):  # For DPAv4
        super().__init__()
        self.noise_factor = round(noise_factor, 1)
        self.noise_scale = round(noise_scale, 2)
        self.noise_level = round(noise_factor * noise_scale, 2)

    def apply_on_trace(self, trace: np.ndarray):
        return trace + np.random.uniform(-self.noise_level, self.noise_level, trace.shape[0])

    def get_cost(self):
        return round(self.noise_factor * 5, 2)

    @classmethod
    def get_instances_within_budget(cls, budget: float, noise_scale: float) -> List['UniformNoise']:
        for i in np.arange(0.1, np.floor(budget * 2)/10 + 0.01, 0.1):
            yield UniformNoise(i, noise_scale)

    @staticmethod
    def countermeasure_id():
        return "uniform_noise"

    def __str__(self):
        return "UniformNoise(noise_factor={:.2f},noise_scale={:.2f})".format(self.noise_factor, self.noise_scale)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.noise_level == other.noise_level

    def __hash__(self):
        return super(UniformNoise, self).__hash__()
