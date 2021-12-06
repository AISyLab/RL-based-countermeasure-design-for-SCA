import random
from typing import List

from aisylab.cython_extensions.countermeasures_cython.random_delay_interrupt import apply_random_delay_interrupts
from . import countermeasure


class RandomDelayInterrupts(countermeasure.Countermeasure):
    """
    Coron et Kizhvatov 2009
    """

    def __init__(self, a, b, rdi_probability, rdi_amplitude):
        super().__init__()
        self.a = a
        self.b = b
        self.probability = round(rdi_probability, 2)
        self.amplitude = round(rdi_amplitude, 2)
        self.cost = RandomDelayInterrupts.calculate_cost(rdi_probability, a, b)

    def apply_on_trace(self, trace):
        return apply_random_delay_interrupts(self.a, self.b, self.probability, self.amplitude, trace)

    def get_cost(self):
        return self.cost

    @staticmethod
    def calculate_cost(rdi_probability, a, b):
        return round(1 + 3 * rdi_probability * (a + b) / 2.0, 2)

    @classmethod
    def get_instances_within_budget(cls, budget: float, noise_scale: float) -> List['RandomDelayInterrupts']:
        a_range = range(1, 10)
        b_range = range(0, 9)
        probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        amplitude = noise_scale  # Amplitude does not have effect on cost
        for a in a_range:
            for b in b_range:
                if b < a:
                    for probability in probability_range:
                        if cls.calculate_cost(probability, a, b) <= budget:
                            yield RandomDelayInterrupts(a, b, probability, amplitude)

    @staticmethod
    def countermeasure_id():
        return "rdi"

    def __str__(self):
        return "RDI(A={:d},B={:d},probability={:.2f},amplitude={:.2f})".format(
            self.a,
            self.b,
            self.probability,
            self.amplitude
        )

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.a == other.a and \
               self.b == other.b and \
               self.probability == other.probability and \
               self.amplitude == other.amplitude

    def __hash__(self):
        return super(RandomDelayInterrupts, self).__hash__()
