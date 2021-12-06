from typing import List

from .countermeasures import countermeasures
from .countermeasures.clock_jitter import ClockJitter
from .countermeasures.countermeasure import Countermeasure
from .countermeasures.desync import Desync
from .countermeasures.random_delay_interrupts import RandomDelayInterrupts
from .countermeasures.uniform_noise import UniformNoise
from .state_enumerator import State


class StateStringUtils(object):
    @staticmethod
    def countermeasures_to_string(state):
        return str('[' + ','.join(map(str, state.countermeasures_applied)) + ']')

    @staticmethod
    def string_to_countermeaures(string) -> List[Countermeasure]:
        parsed_list = countermeasures.parse('countermeasures', string)
        cm_list = []
        for countermeasure in parsed_list:
            if countermeasure[0] == 'jitter':
                cm_list.append(ClockJitter(jitters_level=countermeasure[1]))
            elif countermeasure[0] == 'desync':
                cm_list.append(Desync(desync_level=countermeasure[1]))
            elif countermeasure[0] == 'rdi':
                cm_list.append(RandomDelayInterrupts(
                    a=countermeasure[1], b=countermeasure[2],
                    rdi_probability=countermeasure[3], rdi_amplitude=countermeasure[4]
                ))
            elif countermeasure[0] == 'uniform':
                cm_list.append(UniformNoise(noise_factor=countermeasure[1], noise_scale=countermeasure[2]))

        return cm_list

    @staticmethod
    def convert_cm_string_to_state_list(string) -> List[State]:
        cm_list = StateStringUtils.string_to_countermeaures(string)

        state_list = []
        for i in range(0, len(cm_list) + 1):
            state_list.append(State(
                countermeasures_applied=tuple(cm_list[:i]),
                countermeasures_cost=round(sum([cm.get_cost() for cm in cm_list[:i]]), 2)
            ))

        state_list.append(
            state_list[-1].termination_state()
        )
        return state_list
