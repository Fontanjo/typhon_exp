#######################
## EXPERIMENT CONFIG ##
#######################
import torch
import sys, os
sys.path.insert(0, os.getcwd())
from experiment import Experiment
from pathlib import Path
import utils
import argparse
import copy
import time
from utils import VAELossMSE, VAELossBCE, VAELossBCE_MSE


cfg = {
    # Get GPU number either from the terminal or from the file name
    'trg_gpu' : sys.argv[-1] if not (sys.argv[-1].endswith('.py') or sys.argv[-1].startswith('-')) else Path(__file__).stem.split('_')[-1],
    'trg_n_cpu' : 8, # how many CPU threads to use
    # Datasets
    'dsets' : ['Phoenix-v5', 'DemonAttack-v5'],
    'trg_dset' : 'Phoenix-v5',
    # Pad and crop to get specific dimension
    # One for each dset, or just one if same for all. None to leave as it is
    'working_sizes' : [None],
    # Whether or not to remove the mode
    'remove_mode' : False,
    # Transfer learning type
    # Either 'sequential' or 'parallel'
    'transfer' : 'parallel',
    # Hyperparams
    'lrates' : {
        # One per each DMs
        'train' : [5e-5, 5e-5],
        'spec' : [1e-5, 1e-5],
        # Frozen is for sequential train only, when training with frozen feature extractor
        'frozen' : [1e-3, 1e-3],
    },
    'dropouts' : {
        # First one for the FE, following for the DMs
        'train' : [0.1, 0.1, 0.1],
        'spec' : [0., 0., 0.],
        'frozen' : [0., 0., 0.],
    },
    'batch_size' : {
        'train' : 8,
        'spec' : 64,
    },
    # Only for training, since in specialization it trains on all batches
    'nb_batches_per_epoch' : 1,
    'epochs' : {
        'train' : 150,
        'spec' : 0,
    },
    'architecture' : 'AE8c',
    # Only for autoencoding. Some loss functions requires mu and logvar as well (in particular for VAEs)
    #  In these cases, make sure the dm returns 3 objects (output, mu, logvar)
    'mu_var_loss': False,
    # One per each DMs
    'loss_functions' : [torch.nn.MSELoss(), torch.nn.MSELoss()],
    # One per each DMs
    'optimizers' : [torch.optim.Adam, torch.optim.Adam],
    # Metrics used to compare models, i.e. which one is the best
    'opt_metrics' : {
        'bootstrap' : 'loss',
            'train' : 'loss',
        'spec' : 'accuracy',
    },
    # Frequency of metrics collection during training ans specialization
    'metrics_freq' : {
        'train': 1,
        'spec': 10,
    },
    # Training task (classification / segmentation / autoencoding)
    'training_task' : 'autoencoding',
    # Paths and filenames
    'dsets_path' : 'datasets_autoencoding',
    'ramdir'     : '/dev/shm', # copying data to RAM once to speed it up
    'out_path' : 'results_atari',
    # Type of initialization. Either 'bootstrap', 'random' or 'load'
    'initialization': 'bootstrap',
    # Number of models to tes tin bootstrap. Ignored if 'initialization' is not 'bootstrap'
    'bootstrap_size' : 2000,
    # Number of images to test in bootstrap. In any case at most |training_dset| + |validation_dset|
    #  For now only implemented for training_task autoencoding
    'bootstrap_images': 160,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baselines', action='store_true')
    args = parser.parse_args()

    if not args.baselines:
        exp = Experiment(cfg)
        exp.main_run()

    # Create baselines
    if args.baselines:
        start = time.time()
        utils.print_time('START BASELINES')
        for i, env in enumerate(cfg['dsets']):
            print(env)
            # Create a copy of the config dict
            new_cfg = copy.deepcopy(cfg)
            # Change dsets
            new_cfg['dsets'] = [env]
            new_cfg['trg_dset'] = env
            # Change training type, to use classical loaded
            new_cfg['transfer'] = 'sequential'
            # Get correct working size, lrate, dropout, loss and optimizer
            for metric in ['working_sizes', 'loss_functions', 'optimizers']:
                # Need to do something only if the values are specific for each dset
                if type(cfg[metric] == list and len(cfg[metric]) > 1):
                    new_cfg[metric] = [cfg[metric][i]]
            for metric in ['train', 'spec', 'frozen']:
                if type(cfg['lrates'][metric] == list and len(cfg['lrates'][metric]) > 1):
                    # Copy only if there is a specific value for each dset
                    if len(cfg['lrates'][metric]) == len(cfg['dsets']):
                        new_cfg['lrates'][metric] = [cfg['lrates'][metric][i]]
                if type(cfg['dropouts'][metric] == list and len(cfg['dropouts'][metric]) > 1):
                    if len(cfg['lrates'][metric]) == len(cfg['dsets']) + 1:
                        new_cfg['dropouts'][metric] = [cfg['dropouts'][metric][0], cfg['dropouts'][metric][i+1]] # Copy dropout for fe as well
            # Change the output path
            new_cfg['out_path'] = f'{cfg["out_path"]}/baselines/{env}'
            # # Put a flag for Experiment, which will create different paths
            # new_cfg['baseline'] = True
            # Create and run exp
            new_exp = Experiment(new_cfg)
            new_exp.main_run()
        utils.print_time(f'END BASELINES')
        stop = time.time()
        print(f'Baselines generated in {round(stop - start), 4}s')
