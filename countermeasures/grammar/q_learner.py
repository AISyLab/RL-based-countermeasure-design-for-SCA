import math
import os

import numpy as np
import pandas as pd

from .countermeasures import countermeasures
from .state_enumerator import State, StateEnumerator
from .state_string_utils import StateStringUtils


class QValues(object):
    """ Stores Q_values with helper functions."""

    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_countermeasures',
                                                                'start_cost',
                                                                'start_terminate',
                                                                'end_countermeasures',
                                                                'end_cost',
                                                                'end_terminate',
                                                                'utility']]):
            start_state = State(
                countermeasures_applied=tuple(StateStringUtils.string_to_countermeaures(row[0])),
                countermeasures_cost=round(row[1], 2),
                terminate=bool(row[2])
            ).as_tuple()
            end_state = State(
                countermeasures_applied=tuple(StateStringUtils.string_to_countermeaures(row[3])),
                countermeasures_cost=round(row[4], 2),
                terminate=bool(row[5])
            ).as_tuple()
            utility = row[6]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

    def to_dataframe(self) -> pd.DataFrame:
        start_countermeasures = []
        start_cost = []
        start_terminate = []
        end_countermeasures = []
        end_cost = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = State(*start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = State(*self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])

                start_countermeasures.append(
                    StateStringUtils.countermeasures_to_string(start_state)
                )
                start_cost.append(start_state.countermeasures_cost)
                start_terminate.append(start_state.terminate)

                end_countermeasures.append(
                    StateStringUtils.countermeasures_to_string(to_state)
                )
                end_cost.append(to_state.countermeasures_cost)
                end_terminate.append(to_state.terminate)

        return pd.DataFrame({'start_countermeasures': start_countermeasures,
                             'start_cost': start_cost,
                             'start_terminate': start_terminate,
                             'end_countermeasures': end_countermeasures,
                             'end_cost': end_cost,
                             'end_terminate': end_terminate,
                             'utility': utility})

    def save_to_csv(self, q_csv_path):
        self.to_dataframe().to_csv(q_csv_path, index=False)


class QLearner:
    """ All Q-Learning updates and policy generator
        Args
            state: The starting state for the QLearning Agent
            q_values: A dictionary of q_values --
                            keys: State tuples (State.as_tuple())
                            values: [state list, qvalue list]
            replay_dictionary: A pandas dataframes
            output_number : number of output neurons
    """

    def __init__(self, hyper_parameters, state_space_parameters, epsilon, state=None, qstore=None,
                 replay_dictionary=pd.DataFrame(columns=[
                     'countermeasures',  # Countermeasures applied
                     'cost',  # Total countermeasures cost
                     'guessing_entropy_at_10_percent',
                     'guessing_entropy_at_50_percent',
                     'guessing_entropy_no_to_0',
                     'ix_q_value_update',  # Iteration for q value update
                     'epsilon',  # For epsilon greedy
                     'time_finished'  # UNIX time
                 ])):

        self.state_list = []

        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters

        # Class that will expand states for us
        self.enum = StateEnumerator(hyper_parameters, state_space_parameters)

        # Starting State
        self.state = State((), 0, False) if not state else state

        # Cached Q-Values -- used for q learning update and transition
        self.qstore = QValues() if not qstore else qstore
        self.replay_dictionary = replay_dictionary

        self.epsilon = epsilon  # epsilon: parameter for epsilon greedy strategy

        self.max_budget = self.state_space_parameters.countermeasures_budget

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

    def generate_countermeasures(self):
        # Have Q-Learning agent sample current policy to generate a list of countermeasures and convert to string format
        self._reset_for_new_walk()
        state_list = self._run_agent()

        cm_string = StateStringUtils.countermeasures_to_string(state_list[-1])

        # Check if we have already trained this model
        if cm_string in self.replay_dictionary['countermeasures'].values:
            (_,
             guessing_entropy_at_10_percent,
             guessing_entropy_at_50_percent,
             guessing_entropy_no_to_0) = self.get_metrics_from_replay(cm_string)
        else:
            guessing_entropy_at_10_percent = 128
            guessing_entropy_at_50_percent = 128
            guessing_entropy_no_to_0 = 255

        return (
            cm_string, state_list[-1].countermeasures_applied, state_list[-1].countermeasures_cost,
            guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0
        )

    def save_q(self, q_file):
        self.qstore.save_to_csv(q_file)

    def _reset_for_new_walk(self):
        """Reset the state for a new random walk"""
        # Starting State
        self.state = State((), 0, False)

        # Architecture String
        self.state_list = [self.state]

    def _run_agent(self):
        """Have Q-Learning agent sample current policy to generate a set of countermeasures"""
        while not self.state.terminate:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        """Updates self.state according to an epsilon-greedy strategy"""
        if self.state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.state, self.qstore.q)

        action_values = self.qstore.q[self.state.as_tuple()]
        # epsilon greedy choice
        if np.random.random() < self.epsilon:
            action = State(*action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(len(action_values['actions'])) if
                             action_values['utilities'][i] == max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = State(*max_actions[np.random.randint(len(max_actions))])

        self.state = action

        self._post_transition_updates()

    def _post_transition_updates(self):
        # State to go in state list
        self.state_list.append(self.state.copy())

    def sample_replay_for_update(self, iteration):
        # Experience replay to update Q-Values
        for i in range(self.state_space_parameters.replay_number):
            cm_string = np.random.choice(self.replay_dictionary['countermeasures'])
            (countermeasures_cost,
             guessing_entropy_at_10_percent,
             guessing_entropy_at_50_percent,
             guessing_entropy_no_to_0) = self.get_metrics_from_replay(cm_string)

            state_list = StateStringUtils.convert_cm_string_to_state_list(cm_string)

            self.update_q_value_sequence(state_list, self.metrics_to_reward(
                countermeasures_cost, guessing_entropy_at_10_percent,
                guessing_entropy_at_50_percent, guessing_entropy_no_to_0
            ), iteration)

    def get_metrics_from_replay(self, cm_string):
        cm_replay = self.replay_dictionary[self.replay_dictionary['countermeasures'] == cm_string]
        countermeasures_cost = cm_replay['cost'].values[0]
        guessing_entropy_at_10_percent = cm_replay['guessing_entropy_at_10_percent'].values[0]
        guessing_entropy_at_50_percent = cm_replay['guessing_entropy_at_50_percent'].values[0]
        guessing_entropy_no_to_0 = cm_replay['guessing_entropy_no_to_0'].values[0]
        return (
            countermeasures_cost, guessing_entropy_at_10_percent,
            guessing_entropy_at_50_percent, guessing_entropy_no_to_0
        )

    def metrics_to_reward(self, countermeasures_cost, ge_at_10_percent, ge_at_50_percent, ge_no_to_0):
        """How to define reward from opposing network (performance) metrics"""
        max_reward = 3

        # Starting reward
        # Possible mutations:
        # - 1.5 deduction ge at 10 and 20 percent
        # - 1 deduction from ge_no_to_0
        # - 0.5 addition from remaining budget if network is unsuccessful
        reward = 2.5

        # R -= 0-1 + 0-0.5
        reward -= (
                (128 - min(ge_at_10_percent, 128)) / 128  # 0-1
                + (128 - min(ge_at_50_percent, 128)) / (128 * 2)  # 0-0.5
        )

        # The network was successful in the key recovery within the set amount of traces
        if ge_no_to_0 is not None and not math.isnan(ge_no_to_0):
            traces_per_attack = self.hyper_parameters.TRACES_PER_ATTACK + 1  # also punish ge of 0 in the max |traces|
            reward -= (traces_per_attack - ge_no_to_0) / traces_per_attack  # R -= 0-1
        else:
            reward += (self.max_budget - countermeasures_cost) / (2 * self.max_budget)

        return reward / max_reward

    def update_q_value_sequence(self, states, termination_reward, iteration):
        """Update all Q-Values for a sequence."""
        self._update_q_value(states[-2], states[-1], termination_reward, iteration)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i + 1], 0, iteration)

    def _update_q_value(self, start_state, to_state, reward, iteration):
        """ Update a single Q-Value for start_state given the state we transitioned to and the reward. """
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        max_over_next_states = max(self.qstore.q[to_state.as_tuple()]['utilities']) if not to_state.terminate else 0

        action_index = actions.index(to_state.as_tuple())
        learning_rate_alpha = 1 / (iteration ** self.state_space_parameters.learning_rate_omega)

        # Q_Learning update rule
        values[action_index] = (  # Q_t+1(s_i,𝑢) =
                values[action_index] +  # Q_t(s_i,𝑢)
                learning_rate_alpha * (  # α
                        reward  # r_t
                        + self.state_space_parameters.discount_factor  # γ
                        * max_over_next_states  # max_{𝑢'∈ 𝒰(s_j)} Q_t(s_j,𝑢')
                        - values[action_index]  # -Q_t(s_i,𝑢)
                )
        )

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
