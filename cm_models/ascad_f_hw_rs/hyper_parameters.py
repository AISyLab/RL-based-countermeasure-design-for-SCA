from sklearn import preprocessing

from . import state_space_parameters as ssp
import countermeasures.data_loader as data_loader
import numpy as np
import tensorflow as tf

MODEL_NAME = 'ASCAD_F_HW_RS'

# Number of output neurons
NUM_CLASSES = 9  # Number of output neurons

# Input Size
INPUT_SIZE = 700

# Batch Queue parameters
TRAIN_BATCH_SIZE = 50  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000  # Number of training examples
NUM_ITER_PER_EPOCH_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000  # Number of validation examples

MAX_EPOCHS = 50  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 5e-3  # The max LR (scaled linearly with number of gpus used)

# Reward small parameter (Based on Zaid et al. 2019)
# This rewards networks smaller than current state of the art (* 2 because of HW model)
MAX_TRAINABLE_PARAMS_FOR_REWARD = 16960 * 2


# Bulk data folder
BULK_ROOT = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/jrijsdijk/rl-paper/ASCAD_F/cm_experiment_hw_rs/'
DATA_ROOT = BULK_ROOT + '../data/'

# Trained model dir
TRAINED_MODEL_DIR = BULK_ROOT + 'trained_models'
DB_FILE = DATA_ROOT + 'ascad-fixed.h5'

(TRAIN_TRACES, TRAIN_LABELS), (ATTACK_TRACES, ATTACK_LABELS), ATTACK_PLAINTEXT = data_loader.load_hd5_hw_model(
    DB_FILE,
    '/Profiling_traces/traces', '/Profiling_traces/labels',
    '/Attack_traces/traces', '/Attack_traces/labels',
    '/Attack_traces/metadata'
)

NOISE_SCALE = data_loader.get_noise_scale(TRAIN_TRACES)

USE_OCLR = True

MODEL_PREPROCESSING = [
    preprocessing.StandardScaler()
]

MODEL_LAYERS = [
    tf.keras.layers.Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same'),
    tf.keras.layers.AveragePooling1D(4, strides=4),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(15, kernel_initializer='he_uniform', activation='selu'),
    tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu'),
    tf.keras.layers.Dense(4, kernel_initializer='he_uniform', activation='selu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
]

# Unmask files
KEY = np.load(DATA_ROOT + 'key.npy')
ATTACK_KEY_BYTE = 2
ATTACK_PRECOMPUTED_BYTE_VALUES = np.array(
    [[bin(x).count("1") for x in row] for row in np.load(DATA_ROOT + f'attack_precomputed_byte{ATTACK_KEY_BYTE}_values.npy')]
)

TRACES_PER_ATTACK = 2000  # Maximum number of traces to use per attack
NUM_ATTACKS = 100  # Number of attacks to average the GE over
