from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from countermeasures.grammar.countermeasures.countermeasure import Countermeasure
from countermeasures.attack.one_cycle_lr import OneCycleLR
from countermeasures.attack import utils


class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.key = self.hp.KEY
        self.labels = self.hp.TRAIN_LABELS
        self.attack_labels = getattr(self.hp, 'ATTACK_LABELS', np.ndarray((0, 0)))
        self.precomputed_byte_values = self.hp.ATTACK_PRECOMPUTED_BYTE_VALUES

        self.split_validation_from_attack = getattr(self.hp, 'VALIDATION_FROM_ATTACK_SET', False)

    def compile_model(self, loss, metric_list):
        _optimizer = tf.keras.optimizers.deserialize({
            'class_name': self.hp.OPTIMIZER, 'config': {'learning_rate': self.hp.MAX_LR}}
        )
        model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.hp.INPUT_SIZE, 1))] + self.hp.MODEL_LAYERS
        )
        model.compile(optimizer=_optimizer, loss=loss, metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    @staticmethod
    def get_strategy():
        return tf.distribute.MirroredStrategy()

    def get_preprocessed_traces(self, countermeasures: List[Countermeasure]):
        features = pd.DataFrame(self.hp.TRAIN_TRACES)
        attack_features = pd.DataFrame(self.hp.ATTACK_TRACES)

        for countermeasure in countermeasures:
            print("\n")
            print("Applying {} to features".format(countermeasure))
            features = countermeasure.apply_on_traces(features)

            print("")
            print("Applying {} to attack_features".format(countermeasure))
            attack_features = countermeasure.apply_on_traces(attack_features)

        print("\nFinished applying countermeasures")

        for preprocessing in self.hp.MODEL_PREPROCESSING:
            features = preprocessing.fit_transform(features)
            attack_features = preprocessing.transform(attack_features)

        if len(self.hp.MODEL_PREPROCESSING) == 0:
            features = features.to_numpy()
            attack_features = attack_features.to_numpy()

        features = features.reshape((features.shape[0], features.shape[1], 1))
        attack_features = attack_features.reshape(
            (attack_features.shape[0], attack_features.shape[1], 1)
        )
        return features, attack_features

    def train_and_predict(self, model, features, attack_features, parallel_no=1):
        features, labels = utils.shuffle_arrays_together(features, self.labels)

        training_features = features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
        training_labels = to_categorical(
            labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], num_classes=self.hp.NUM_CLASSES
        )

        if self.split_validation_from_attack:
            validation_features, validation_labels = utils.shuffle_arrays_together(
                attack_features, self.attack_labels
            )
            validation_features = validation_features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL]
            validation_labels = to_categorical(
                self.attack_labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL],
                num_classes=self.hp.NUM_CLASSES
            )
        else:
            validation_features = features[self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:]
            validation_labels = to_categorical(
                labels[self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:], num_classes=self.hp.NUM_CLASSES
            )

        callbacks = []
        if self.hp.USE_OCLR:
            callbacks.append(OneCycleLR(
                max_lr=self.hp.MAX_LR * parallel_no, end_percentage=0.2, scale_percentage=0.1,
                maximum_momentum=None,
                minimum_momentum=None, verbose=True
            ))

        model.fit(
            x=training_features, y=training_labels, epochs=self.hp.MAX_EPOCHS,
            batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no, verbose=2,
            validation_data=(validation_features, validation_labels), shuffle=True, callbacks=callbacks
        )

        return model.predict(attack_features), \
            model.evaluate(x=validation_features, y=validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE)

    def perform_attacks(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=True, save_graph=save_graph, filename=filename, folder=folder
        )

    def perform_attacks_parallel(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n_parallel(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=True, save_graph=save_graph, filename=filename, folder=folder
        )


if __name__ == '__main__':
    from cm_models.ches_ctf_hw import hyper_parameters as hp
    import countermeasures.grammar.countermeasures as cm
    import time

    tf_runner = TensorFlowRunner(state_space_parameters=hp.ssp, hyper_parameters=hp)
    strategy = tf_runner.get_strategy()

    with strategy.scope():
        # countermeasures = [cm.clock_jitter.ClockJitter(jitters_level=2)]
        # countermeasures = [cm.desync.Desync(desync_level=10)]
        # countermeasures = [cm.uniform_noise.UniformNoise(0.10, hp.NOISE_SCALE)]
        # countermeasures = [cm.random_delay_interrupts.RandomDelayInterrupts(a=2, b=1, rdi_probability=0.2, rdi_amplitude=hp.NOISE_SCALE)]
        countermeasures = []
        start_cm = time.time()
        traces, attack_traces = tf_runner.get_preprocessed_traces(countermeasures)
        end_cm = time.time()

        tf_model = tf_runner.compile_model(loss='categorical_crossentropy', metric_list=['accuracy'])
        tf_model.summary()
        # tf.keras.utils.plot_model(tf_model, to_file='model_test.dot', show_shapes=True, show_layer_names=False)
        # exit(0)

        start_model = time.time()
        model_predictions, metrics = tf_runner.train_and_predict(
            tf_model, traces, attack_traces, strategy.num_replicas_in_sync
        )
        end_model = time.time()

    print("Performing the attacks:")
    start_ge = time.time()
    guessing_entropy = tf_runner.perform_attacks_parallel(model_predictions, True, hp.MODEL_NAME, "data/")
    end_ge = time.time()

    print("\n t_GE = ")
    print(np.array2string(
        guessing_entropy,
        formatter={"float_kind": lambda x: f"{x:g}"},
        threshold=hp.TRACES_PER_ATTACK + 1  # Always print full array instead of summary
    ))
    ge_no_to_0 = np.where(guessing_entropy <= 0)
    print("\n mean GE == 0 with #traces: {}".format(ge_no_to_0[0][0] + 1 if len(ge_no_to_0[0] > 0) else "∞"))
    print("\nTimings:")
    print(f"\t- Countermeasure preprocessing: {int(end_cm - start_cm)}s")
    print(f"\t- Model training: {int(end_model - start_model)}s")
    print(f"\t- GE calculation: {int(end_ge - start_ge)}s")
    exit(0)
