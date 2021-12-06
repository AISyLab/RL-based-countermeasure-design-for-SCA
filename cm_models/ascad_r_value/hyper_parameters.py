from sklearn import preprocessing

from . import state_space_parameters as ssp
import countermeasures.data_loader as data_loader
import numpy as np
import tensorflow as tf

MODEL_NAME = 'ASCAD_R_Value'

# Number of output neurons
NUM_CLASSES = 256  # Number of output neurons

# Input Size
INPUT_SIZE = 1400

# Batch Queue parameters
TRAIN_BATCH_SIZE = 400  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100_000  # Number of training examples
VALIDATION_FROM_ATTACK_SET = True
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000  # Number of validation examples

MAX_EPOCHS = 50  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 5e-3  # The max LR (scaled linearly with number of gpus used)

# Bulk data folder
BULK_ROOT = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/jrijsdijk/rl-paper/ASCAD_R/cm_experiment_value/'
DATA_ROOT = BULK_ROOT + '../data/'

# Trained model dir
TRAINED_MODEL_DIR = BULK_ROOT + 'trained_models'
DB_FILE = DATA_ROOT + 'ascad-variable.h5'

(TRAIN_TRACES, TRAIN_LABELS), (ATTACK_TRACES, ATTACK_LABELS), ATTACK_PLAINTEXT = data_loader.load_hd5(
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
    tf.keras.layers.Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same'),
    tf.keras.layers.AveragePooling1D(75, strides=75),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(30, kernel_initializer='he_uniform', activation='selu'),
    tf.keras.layers.Dense(2, kernel_initializer='he_uniform', activation='selu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
]

KEY = np.load(DATA_ROOT + 'attack_key.npy')
ATTACK_KEY_BYTE = 2
ATTACK_PRECOMPUTED_BYTE_VALUES = np.load(DATA_ROOT + f'attack_precomputed_byte{ATTACK_KEY_BYTE}_values.npy')

TRACES_PER_ATTACK = 2000  # Maximum number of traces to use per attack
NUM_ATTACKS = 20  # Number of attacks to average the GE over
