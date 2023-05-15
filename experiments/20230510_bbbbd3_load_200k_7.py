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


cfg = {
    # Get GPU number either from the terminal or from the file name
    'trg_gpu' : sys.argv[-1] if not (sys.argv[-1].endswith('.py') or sys.argv[-1].startswith('-')) else Path(__file__).stem.split('_')[-1],
    'trg_n_cpu' : 8, # how many CPU threads to use
    # Datasets
    'dsets' : ['BUS_SELECTED', 'LINK_BRATS_LGG_flair',  'BUSI', 'BRAIN'],
    'trg_dset' : 'BUS_SELECTED',
    # Pad and crop to get specific dimension
    # One for each dset, or just one if same for all. None to leave as it is
    'working_sizes': [(512, 512), (256, 256), (512, 512), (256, 256)],
    # Whether or not to remove the mode
    'remove_mode' : False, # Only used for task 'autoencoding'
    # Transfer learning type
    # Either 'sequential' or 'parallel'
    'transfer' : 'parallel',
    # Hyperparams
    'lrates' : {
        # One per each DMs
        'train' : [2e-4]*4,
        'spec' : [2e-5]*4,
       # Frozen is for sequential train only, when training with frozen feature extractor
        'frozen' : [1e-5]*4,
    },
    'dropouts' : {
        # First one for the FE, following for the DMs
        'train' : [0.1]*5,
        'spec' : [0.1]*5,
        'frozen' : [0.]*5,
    },
    'batch_size' : {
        'train' : 8,
        'spec' : 8,
    },
    # Only for training, since in specialization it trains on all batches
    'nb_batches_per_epoch' : 1,
    'epochs' : {


        # 'train' : 50000,


        'train' : 100000,
        'spec' : 0,
    },
    'architecture' : 'paper_segmentation_v3', # Pass only top 32, to force learning bigger sprites
    # 'architecture': 'paper_segmentation',
    # Only for autoencoding. Some loss functions requires mu and logvar as well (in particular for VAEs)
    #  In these cases, make sure the dm returns 3 objects (output, mu, logvar)
    'mu_var_loss': False,
    # One per each DMs, or just one to be copied
    'loss_functions' : [utils.DiceLoss()],
    # 'loss_functions': [torch.nn.CrossEntropyLoss()],
    # One per each DMs, or just one to be copied
    'optimizers' : [torch.optim.Adam],
    # Metrics used to compare models, i.e. which one is the best
    'opt_metrics' : {
        'bootstrap' : 'dice',
            'train' : 'dice',
        'spec' : 'dice',
    },
    # Frequency of metrics collection during training ans specialization


    # HOW OFTEN YOU TEST THE ENTIRE DATASET TO SAVE METRICS "every 2"
    'metrics_freq' : {
        'train': 250,
        'spec': 10,
    },


    # Training task (classification / segmentation / autoencoding)
    'training_task' : 'segmentation',
    # Paths and filenames
    'dsets_path' : 'datasets',
    'ramdir'     : '/dev/shm', # copying data to RAM once to speed it up
    'out_path' : 'results',


    # Type of initialization. Either 'bootstrap', 'random' or 'load'
    # 'initialization': 'random',
    'initialization': 'load',


    # Number of models to tes tin bootstrap. Ignored if 'initialization' is not 'bootstrap'
    # 'bootstrap_size' : 2000,
    'bootstrap_size' : 300,




    # ONLY IMPLEMENTED FOR AUTOENCODERS, WON'T WORK OTHERWISE
    # Number of images to test in bootstrap. In any case at most |training_dset| + |validation_dset|
    #  For now only implemented for training_task autoencoding
    'bootstrap_images': 400,



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
    # DEBUG: uncaught exceptions drop you into ipdb for postmortem debugging
    import sys, IPython; sys.excepthook = IPython.core.ultratb.ColorTB(call_pdb=True)
    exp.main_run()
