#######################
## EXPERIMENT CONFIG ##
#######################
import torch
import sys, os
sys.path.insert(0, os.getcwd())
from experiment import Experiment
from pathlib import Path
import utils


cfg = {
    # Get GPU number either from the terminal or from the file name
    'trg_gpu' : sys.argv[-1] if not sys.argv[-1].endswith('.py') else Path(__file__).stem.split('_')[-1],
    'trg_n_cpu' : 8, # how many CPU threads to use
    # Datasets
    'dsets' : ['Seaquest-v5', 'SpaceInvaders-v5'],
    'trg_dset' : 'Seaquest-v5',
    # Pad and crop to get specific dimension
    # One for each dset, or just one if same for all. None to leave as it is
    'working_sizes' : [(256, 256)],
    # Transfer learning type
    # Either 'sequential' or 'parallel'
    'transfer' : 'parallel',
    # Hyperparams
    'lrates' : {
        # One per each DMs
        'train' : [1e-2],
        'spec' : [8e-10],
        # Frozen is for sequential train only, when training with frozen feature extractor
        'frozen' : [1e-7],
    },
    'dropouts' : {
        # First one for the FE, following for the DMs
        'train' : [0., 0.],
        'spec' : [0., 0.],
        'frozen' : [0., 0.],
    },
    'batch_size' : {
        'train' : 4,
        'spec' : 64,
    },
    # Only for training, since in specialization it trains on all batches
    'nb_batches_per_epoch' : 1,
    'epochs' : {
        'train' : 300,
        'spec' : 0,
    },
    'architecture' : 'AE2Long',
    # One per each DMs
    'loss_functions' : [torch.nn.MSELoss()],
    # One per each DMs
    'optimizers' : [torch.optim.Adam],
    # Metrics used to compare models, i.e. which one is the best
    'opt_metrics' : {
        'bootstrap' : 'iou',
            'train' : 'iou',
        'spec' : 'accuracy',
    },
    # Frequency of metrics collection during training
    'metrics_freq' : 10,
    # Training task (classification / segmentation / autoencoding)
    'training_task' : 'autoencoding',
    # Paths and filenames
    'dsets_path' : 'datasets_autoencoding',
    'ramdir'     : '/dev/shm', # copying data to RAM once to speed it up
    'out_path' : 'results_atari',
    # Type of initialization. Either 'bootstrap', 'random' or 'load'
    'initialization': 'random',
    # bootstrap. Ignored if 'initialization' is not 'bootstrap'
    'bootstrap_size' : 0,
    # Add timestamp to avoid overwriting on the folder (e.g. if we want to repeat the same exp)
    'timestamp' : False,
    # If we want to resume the current exp (False for a new experiment)
    # Makes only sense for typhon training
        'resume' : False,
    # Experiment name
    'exp_file' : __file__,
}
#######################

if __name__ == '__main__':
    exp = Experiment(cfg)
    exp.main_run()