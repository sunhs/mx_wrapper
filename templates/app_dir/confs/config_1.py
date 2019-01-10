import os
import pickle

######################################################################
# Needed by my_modules.
######################################################################
# Names and paths.
PROJ_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'
)
APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(PROJ_ROOT_DIR, 'data')
MODEL_DIR = os.path.join(
    APP_DIR, 'save_dir',
    'model_{}'.format(os.path.basename(__file__).rsplit('.')[0].split('_')[1])
)
PRETRAIN_PATH = ''
PARAM_DIR = os.path.join(MODEL_DIR, 'states')
PARAM_PREFIX = 'epoch'
PARAM_INDEX = None
SAVE_EPOCH_FREQ = 0

# Hypers and devices.
MAX_EPOCHS = 50
BATCH_SIZE = {'train': 32, 'test': 1}
OPTIMIZER = 'adam'
OPT_PARAMS = {
    'learning_rate': 1e-3,
    'wd': 5e-4
}
PARAM_GROUPS = [
    {
        'params': '.*',
        'lr_mult': 0.1,
        'wd_mult': 0.1
    }, {
        'params': ['.*dense'],
        'lr_mult': 1,
        'wd_mult': 1
    }
]
GPUS = []
NUM_WORKERS = 8

if not os.path.exists(PARAM_DIR):
    os.makedirs(PARAM_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

with open(os.path.join(MODEL_DIR, 'log.pkl'), 'wb') as f:
    log = {'train': {'loss': []}, 'test': {'loss': []}}
    pickle.dump(log, f)

with open(os.path.join(MODEL_DIR, 'train_log.pkl'), 'wb') as f:
    train_log = []
    pickle.dump(train_log, f)
