import argparse
import os
from typing import Generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .grammar import q_learner


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(results_dir: str, input_size: int, traces_per_attack: int):
    qstore = q_learner.QValues()
    qstore.load_q_values(os.path.join(results_dir, 'qlearner_logs', 'q_values.csv'))
    replay_dic = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv')).drop_duplicates(
        subset='countermeasures', keep='first')
    replay_dic_last = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv')).drop_duplicates(
        subset='countermeasures', keep='last')
    ssp = AttrDict({
        'input_size': input_size,
        'output_states': 256,
        'init_utility': 0.3,
        'countermeasures_budget': 5,
    })

    ql = q_learner.QLearner(AttrDict({'ssp': ssp, 'TRACES_PER_ATTACK': traces_per_attack}),
                            ssp,
                            0.0,
                            qstore=qstore,
                            replay_dictionary=replay_dic)

    metrics = ["cost", "GE at 10% traces", "GE at 50% traces", "GE #traces to 0", "reward"]
    (cm_string,
     countermeasures,
     cost,
     guessing_entropy_at_10_percent,
     guessing_entropy_at_50_percent,
     guessing_entropy_no_to_0) = ql.generate_countermeasures()
    reward = ql.metrics_to_reward(
        cost, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0
    )
    iteration = replay_dic[replay_dic['countermeasures'] == cm_string]['ix_q_value_update'].values[0]

    replay_dic['reward'] = replay_dic.apply(
        lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['countermeasures'])),
        axis='columns'
    )
    results_sorted = replay_dic.sort_values(by=['reward'], ascending=False)
    title = os.path.join(*results_dir.split(os.path.sep)[-2:])

    with open(os.path.join(results_dir, 'results_overview.txt'), mode="w") as file:
        file.write(f"Results for {os.path.join(*results_dir.split(os.path.sep)[-2:])}\n\n")

        file.write("Best countermeasures according to Q-Learning:\n")
        file.write(f"{cm_string}\n")
        file.write(f"First found at iteration: {iteration}\n")
        file.write("Metrics:\n")
        file.writelines(
            iterable_as_list(metrics, [
                cost, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0, reward
            ])
        )

        q_values = ql.qstore.to_dataframe()
        file.write(f"\n\nAverage q_value: {q_values['utility'].mean()}\n")
        file.write(
            f"Average (filtered) q_value: {q_values[q_values['utility'] != ssp.init_utility]['utility'].mean()}\n")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000,
                               'display.max_colwidth', 200):
            file.write("\n\nTop 20 total reward countermeasures:\n")
            file.write(str(results_sorted.head(20)))
            file.write("\n\nBottom 20 total reward countermeasures:\n")
            file.write(str(results_sorted.tail(20)))

    plt.style.use(os.path.dirname(__file__)+'/scatter_plot.mplstyle')
    replay_dic.plot.scatter(x='reward', y='cost', c='epsilon', colormap='viridis',
                              figsize=(10, 9), xlim=(-0.02, 1.02), ylim=(-0.1, 5.1))
                              
    plt.xlabel('Q-Learning reward')
    plt.ylabel('Relative Countermeasure Cost')
    ax = plt.gca()
    ax.figure.axes[-1].set_ylabel('Epsilon When First Generated')
    max_beaten = replay_dic.loc[replay_dic['guessing_entropy_no_to_0'].idxmax()]
    ax.figure.axes[0].axvline(max_beaten['reward'], color='red', lw=1.0)
    plt.savefig(
        os.path.join(results_dir, f'{title.replace(os.path.sep, "_")}_first_scatter.svg'),
        format='svg', dpi=150, bbox_inches='tight'
    )
    plt.close()

    replay_dic_last['reward'] = replay_dic_last.apply(
        lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['countermeasures'])),
        axis='columns'
    )

    plt.style.use(os.path.dirname(__file__)+'/scatter_plot.mplstyle')
    replay_dic_last.plot.scatter(x='reward', y='cost', c='epsilon', colormap='viridis',
                            figsize=(10, 9), xlim=(-0.02, 1.02), ylim=(-0.1, 5.1))
    plt.xlabel('Q-Learning reward')
    plt.ylabel('Relative Countermeasure Cost')
    ax = plt.gca()
    ax.figure.axes[-1].set_ylabel('Epsilon When Last Generated')
    plt.savefig(
        os.path.join(results_dir, f'{title.replace(os.path.sep, "_")}_last_scatter.svg'),
        format='svg', dpi=150, bbox_inches='tight'
    )
    plt.close()

    return replay_dic


def iterable_as_list(descriptions: iter, dictionary: iter) -> Generator[str, str, None]:
    for description, el in zip(descriptions, dictionary):
        yield f"- {description}: {el}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'results_dir',
        help='Directory with results of an experiment'
    )
    parser.add_argument(
        'input_size',
        help='The input layer size',
        default=700,
        type=int
    )
    parser.add_argument(
        'traces_per_attack',
        help='The number of traces used per attack',
        type=int
    )
    args = parser.parse_args()

    subdirs = next(os.walk(args.results_dir))[1]
    if np.isin(subdirs, ["graphs", "trained_models", "qlearner_logs"]).all():
        results = main(args.results_dir, args.input_size, args.traces_per_attack)
    else:
        print(f"Results dir {args.results_dir} does not contain the required graphs, trained_models and qlearner_logs "
              "subfolders")
