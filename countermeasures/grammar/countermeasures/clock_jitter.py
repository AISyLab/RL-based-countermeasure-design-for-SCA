from typing import List

import numpy as np
from aisylab.cython_extensions.countermeasures_cython.clock_jitter import apply_clock_jitters
from . import countermeasure


class ClockJitter(countermeasure.Countermeasure):
    def __init__(self, jitters_level):
        super().__init__()
        self.jitters_level = jitters_level // 2  # jitters_level can go on either side of 0, so 1/2 of amplitude arg

    def apply_on_trace(self, trace: np.ndarray):
        return apply_clock_jitters(self.jitters_level, trace)

    def get_cost(self):
        return round(self.jitters_level / .8, 2)

    @classmethod
    def get_instances_within_budget(cls, budget: float, noise_scale: float) -> List['ClockJitter']:
        for i in range(2, int(budget * 1.6) + 1, 2):
            yield ClockJitter(jitters_level=i)

    @staticmethod
    def countermeasure_id():
        return "clock_jitter"

    def __str__(self):
        return "ClockJitter(jitters_level={:d})".format(
            self.jitters_level * 2
        )

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.jitters_level == other.jitters_level

    def __hash__(self):
        return super(ClockJitter, self).__hash__()

