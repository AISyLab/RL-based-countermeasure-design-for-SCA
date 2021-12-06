import random
from typing import List

from . import countermeasure
from scipy.ndimage.interpolation import shift


class Desync(countermeasure.Countermeasure):
    def __init__(self, desync_level):
        super().__init__()
        self.desync_level = desync_level

    def apply_on_trace(self, trace):
        desync = random.randint(0, self.desync_level)
        return shift(input=trace, shift=-desync, cval=0.0)

    def get_cost(self):
        return round(self.desync_level / 10, 2)

    @classmethod
    def get_instances_within_budget(cls, budget: float, noise_scale: float) -> List['Desync']:
        for i in range(5, int(budget * 10), 5):
            yield Desync(i)

    @staticmethod
    def countermeasure_id():
        return "desync"

    def __str__(self):
        return "Desync(desync_level={:d})".format(self.desync_level)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.desync_level == other.desync_level

    def __hash__(self):
        return super(Desync, self).__hash__()
