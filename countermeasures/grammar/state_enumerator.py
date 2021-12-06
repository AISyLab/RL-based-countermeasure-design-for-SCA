from typing import List, Tuple

from ..grammar.countermeasures.countermeasure import Countermeasure
from ..grammar.countermeasures import COUNTERMEASURES


class State(object):
    def __init__(self,
                 countermeasures_applied: Tuple[Countermeasure, ...] = (),  # Current countermeasures applied
                 countermeasures_cost: float = 0,
                 terminate: bool = False):  # can be constructed from a list instead, list takes precedent
        self.countermeasures_applied = countermeasures_applied
        self.countermeasures_set = set([cm.countermeasure_id() for cm in self.countermeasures_applied])
        self.countermeasures_cost = countermeasures_cost
        self.terminate = terminate

    def __str__(self):
        return str(self.as_tuple())

    def as_tuple(self):
        return self.countermeasures_applied, self.countermeasures_cost, self.terminate

    def as_list(self):
        return list(self.as_tuple())

    def copy(self):
        return State(self.countermeasures_applied, self.countermeasures_cost, self.terminate)

    def termination_state(self):
        return State(self.countermeasures_applied, self.countermeasures_cost, True)


class StateEnumerator(object):
    """Class that deals with:
            Enumerating States (defining their possible transitions)
    """

    def __init__(self, hyper_parameters, state_space_parameters):
        self.countermeasures = COUNTERMEASURES
        self.countermeasures_set = set(self.countermeasures.keys())
        self.hp = hyper_parameters
        self.ssp = state_space_parameters
        self.max_budget = state_space_parameters.countermeasures_budget
        # Limits
        pass

    def enumerate_state(self, state: State, q_values):
        """Defines all state transitions, populates q_values where actions are valid
        Updates: q_values and returns q_values
        """
        if state.terminate:
            return q_values

        actions = []
        countermeasures_available = self.countermeasures_set - state.countermeasures_set
        for countermeasure in countermeasures_available:
            actions += [
                State(
                    tuple(list(state.countermeasures_applied) + [countermeasure_instance]),
                    round(state.countermeasures_cost + countermeasure_instance.get_cost(), 2)
                )
                for countermeasure_instance
                in COUNTERMEASURES[countermeasure].get_instances_within_budget(
                    self.max_budget - round(state.countermeasures_cost, 2),
                    self.hp.NOISE_SCALE  # Only gets used by countermeasures it is relevant for
                )
            ]

        actions += [state.termination_state()]

        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {
            'actions': [to_state.as_tuple() for to_state in actions],
            'utilities': [self.ssp.init_utility for _ in range(len(actions))]
        }
        return q_values
