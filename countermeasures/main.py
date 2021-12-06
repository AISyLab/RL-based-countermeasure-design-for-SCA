import argparse
import math
import multiprocessing as mp
import os
import contextlib
import signal
import sys
import time
import traceback
from datetime import datetime
from os import path

import cloudpickle
import numpy as np
import pandas as pd

from .attack.tensorflow_runner import TensorFlowRunner
from countermeasures.grammar import q_learner


class TermColors(object):
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class QCoordinator(object):
    def __init__(self,
                 list_path,
                 state_space_parameters,
                 hyper_parameters,
                 epsilon=None,
                 number_models=None,
                 hpc=False):

        print("\n\nRun started at: {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

        self.replay_columns = [
            'countermeasures',  # Countermeasures applied
            'cost',  # Total countermeasures cost
            'guessing_entropy_at_10_percent',
            'guessing_entropy_at_50_percent',
            'guessing_entropy_no_to_0',
            'ix_q_value_update',  # Iteration for q value update
            'epsilon',  # For epsilon greedy
            'time_finished'  # UNIX time
        ]

        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters
        self.hpc = hpc

        self.number_q_updates_per_train = 100

        self.budget = self.state_space_parameters.countermeasures_budget

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.tf_runner = TensorFlowRunner(state_space_parameters, hyper_parameters)
        self.ten_percent_index = self.hyper_parameters.TRACES_PER_ATTACK // 10 - 1
        self.fifty_percent_index = self.hyper_parameters.TRACES_PER_ATTACK // 2 - 1

        while not self.check_reached_limit():
            self.q_learning_step()

        print('{}{}Experiment Complete{}'.format(TermColors.BOLD, TermColors.OKGREEN, TermColors.RESET))
        sys.exit(0)

    def q_learning_step(self):
        cm_string, countermeasures, cost, iteration = self.generate_new_countermeasures()
        print('\n{}Using countermeasures:\n{}\nIteration {:d}, Epsilon {:f}: [{:d}/{:d}]{}'.format(
            TermColors.OKBLUE, cm_string, iteration, self.epsilon, self.number_unique(self.epsilon),
            self.number_models, TermColors.RESET
        ))

        parent, child = mp.Pipe(duplex=False)
        process = mp.Process(target=self._train_and_predict, args=(
            cloudpickle.dumps(self.tf_runner),
            cloudpickle.dumps(countermeasures),
            child
        ))

        process.start()
        pool = _busy_wait() if self.hpc else None
        predictions, (test_loss, test_accuracy) = cloudpickle.loads(parent.recv())
        process.join()
        if self.hpc:
            _stop_busy_wait(pool)

        guessing_entropy = self.tf_runner.perform_attacks_parallel(
            predictions, save_graph=True, filename=f"{self.hyper_parameters.MODEL_NAME}_{iteration:04}",
            folder=f"{self.hyper_parameters.BULK_ROOT}/graphs"
        )

        ge_no_to_0 = np.where(guessing_entropy <= 0)[0]

        self.incorporate_countermeasures(
            cm_string, cost, guessing_entropy[self.ten_percent_index],
            guessing_entropy[self.fifty_percent_index], ge_no_to_0[0] if len(ge_no_to_0) > 0 else None,
            float(self.epsilon), [iteration]
        )

    @staticmethod
    def _train_and_predict(tf_runner, countermeasures, return_pipe):
        tf_runner = cloudpickle.loads(tf_runner)
        countermeasures = cloudpickle.loads(countermeasures)
        strategy = tf_runner.get_strategy()
        parallel_no = strategy.num_replicas_in_sync
        if parallel_no is None:
            parallel_no = 1

        with strategy.scope():
            model = tf_runner.compile_model(loss='categorical_crossentropy', metric_list=['accuracy'])

            features, attack_features = tf_runner.get_preprocessed_traces(countermeasures)

            return_pipe.send(cloudpickle.dumps(
                tf_runner.train_and_predict(model, features, attack_features, parallel_no),
            ))

    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print('Found replay dictionary')
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print('Found q values')
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None

        ql = q_learner.QLearner(self.hyper_parameters,
                                self.state_space_parameters,
                                self.epsilon,
                                qstore=qstore,
                                replay_dictionary=self.replay_dictionary)

        return ql

    @staticmethod
    def filter_replay_for_first_run(replay):
        """ Order replay by iteration, then remove duplicate countermeasures keeping the first"""
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['countermeasures'])

    def number_unique(self, epsilon=None):
        """Epsilon defaults to the minimum"""
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        """ Returns True if the experiment is complete"""
        if len(self.replay_dictionary):
            completed_current = self.number_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break

                else:
                    completed_experiment = True

                return completed_experiment

            else:
                return False

    def generate_new_countermeasures(self):
        try:
            (cm_string,
             countermeasures,
             cost,
             guessing_entropy_at_10_percent,
             guessing_entropy_at_50_percent,
             guessing_entropy_no_to_0,) = self.qlearner.generate_countermeasures()

            # We have already tried this countermeasure combination
            while cm_string in self.replay_dictionary.countermeasures.values:
                self.q_training_step += 1
                self.incorporate_countermeasures(
                    cm_string,
                    cost,
                    guessing_entropy_at_10_percent,
                    guessing_entropy_at_50_percent,
                    guessing_entropy_no_to_0,
                    self.epsilon,
                    [self.q_training_step]
                )
                # Attempt to generate another countermeasure combination
                (cm_string,
                 countermeasures,
                 cost,
                 guessing_entropy_at_10_percent,
                 guessing_entropy_at_50_percent,
                 guessing_entropy_no_to_0,) = self.qlearner.generate_countermeasures()

            # We have found a countermeasure combination we have not tried yet
            self.q_training_step += 1
            return cm_string, countermeasures, cost, self.q_training_step

        except Exception:
            print("Exception occurred when generating new countermeasures:")
            print(traceback.print_exc())
            sys.exit(1)

    def incorporate_countermeasures(self, cm_string, cost, ge_at_10_percent, ge_at_50_percent, ge_no_to_0,
                                    epsilon, iterations):
        try:
            # If we sampled the same countermeasures many times, we should add each of them into the replay database
            for iteration in iterations:
                self.replay_dictionary = pd.concat([
                    self.replay_dictionary,
                    pd.DataFrame({
                        'countermeasures': [cm_string],
                        'cost': [cost],
                        'guessing_entropy_at_10_percent': [ge_at_10_percent],
                        'guessing_entropy_at_50_percent': [ge_at_50_percent],
                        'guessing_entropy_no_to_0': [ge_no_to_0],
                        'ix_q_value_update': [iteration],
                        'epsilon': [epsilon],
                        'time_finished': [time.time()]
                    })
                ])
                with atomic_overwrite(self.replay_dictionary_path) as out_file:
                    self.replay_dictionary.to_csv(out_file, index=False, columns=self.replay_columns)
            print(0)
            self.qlearner.update_replay_database(self.replay_dictionary)
            print(1)

            for iteration in iterations:
                self.qlearner.sample_replay_for_update(iteration)
            print(2)

            with atomic_overwrite(os.path.join(self.list_path, 'q_values.csv')) as out_file:
                self.qlearner.save_q(out_file)
            print(3)
            if ge_no_to_0 is None or math.isnan(ge_no_to_0):
                ge_no_to_0 = "∞"
            print(4)
            print('{}Incorporated countermeasures, ge at 10%: {}, ge at 50%: {}, t_GE <= 0: {}:\n{}{}'.format(
                TermColors.YELLOW, ge_at_10_percent, ge_at_50_percent, ge_no_to_0, cm_string, TermColors.RESET
            ))
            print(5)
        except Exception:
            print("Exception occurred when incorporating countermeasures:")
            print(traceback.print_exc())
            print(Exception)
            print('{}Incorporated countermeasures, ge at 10%: {}, ge at 50%: {}, t_GE <= 0: {}:\n{}{}'.format(
                TermColors.YELLOW, ge_at_10_percent, ge_at_50_percent, ge_no_to_0, cm_string, TermColors.RESET
            ))


@contextlib.contextmanager
def atomic_overwrite(filename):
    temp = filename + '~'
    with open(temp, "w") as f:
        yield f
    os.rename(temp, filename)  # this will only happen if no exception was raised


stop_loop = mp.Value('b', 0)


def _busy_wait_f(x):
    global stop_loop
    while stop_loop.value == 0:
        x * x
    sys.exit(0)


def _busy_wait():
    global stop_loop
    stop_loop.value = 0

    thread_count = len(os.sched_getaffinity(0)) if 'sched_getaffinity' in dir(os) else mp.cpu_count()
    processes = max(1, thread_count - 4)
    pool = mp.Pool(processes)
    pool.map_async(_busy_wait_f, range(processes))

    signal.signal(signal.SIGINT, lambda x, y: _stop_busy_wait(pool))
    signal.signal(signal.SIGTERM, lambda x, y: _stop_busy_wait(pool))

    return pool


def _stop_busy_wait(pool):
    import threading
    global stop_loop
    stop_loop.value = 1
    stop_pool = threading.Thread(target=_stop_pool, args=[pool])
    stop_pool.daemon = True
    stop_pool.start()


def _stop_pool(pool):
    pool.close()
    pool.terminate()


def main():
    parser = argparse.ArgumentParser()

    model_pkgpath = 'cm_models'
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument(
        'model',
        help='Model package name. Package should have a hyper_parameters.py and a state_space_parameters.py file.',
        choices=model_choices
    )

    parser.add_argument('-p', '--hpc', action='store_true')
    parser.add_argument('-eps', '--epsilon', help='For Epsilon Greedy Strategy', type=float)
    parser.add_argument('-nmt', '--number_models_to_train', type=int,
                        help='How many models for this epsilon do you want to train.')

    args = parser.parse_args()

    _model = __import__(
        'cm_models.' + args.model,
        globals(),
        locals(),
        ['state_space_parameters', 'hyper_parameters'],
        0
    )

    factory = QCoordinator(
        path.normpath(path.join(_model.hyper_parameters.BULK_ROOT, "qlearner_logs")),
        _model.state_space_parameters,
        _model.hyper_parameters,
        args.epsilon,
        args.number_models_to_train,
        args.hpc
    )


if __name__ == '__main__':
    main()
