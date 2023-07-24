###########################################
## THIS IS THE GENERIC EXPERIMENT SCRIPT ##
###########################################
# Check folder 'experiments/' to find the actual exps
if __name__ == '__main__':
    print("You should not call this directly. Check folder `experiments`.")
    import sys
    sys.exit()


import os
import datetime
import time
from pathlib import Path
import shutil
import torch
from brutelogger import BruteLogger
import typhon
import utils
import copy


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        assert (not self.cfg['resume']) or (not self.cfg['timestamp']), "Cannot resume experiment with timestamp activated"
        assert (not self.cfg['transfer'] == 'sequential') or (not self.cfg['resume']), "Cannot resume training on sequential"

        self.make_paths()

        # Setup logger
        BruteLogger.save_stdout_to_file(path=self.paths['logs'])

        # Resolve CPU threads and cuda device
        torch.set_num_threads(self.cfg['trg_n_cpu'])

        if torch.cuda.is_available():
            # Need to have a GPU and to precise it at the end of the experiment file name
            # Or in the terminal after the file name
            # Assertion blocks if we cannot cast to int, i.e. last part of experiment file name is not an int
            assert isinstance(int(self.cfg['trg_gpu']), int), "Please precise your GPU at the end of the experiment file name"
            # Will anyway stop if the index is not available or wrong
            self.cuda_device = f"cuda:{int(self.cfg['trg_gpu'])}"
        # Otherwise just go with CPU
        else:
            self.cuda_device = 'cpu'
            torch.set_num_threads(self.cfg['trg_n_cpu'])

        # Give a dropout, learning rate, optimizer and loss function specific to each DMs
        self.dropouts = {}
        self.lrates = {}
        for type in self.cfg['dropouts'].keys():
            self.dropouts[type] =  [self.cfg['dropouts'][type][0], {name:dropout for name, dropout in zip(self.cfg['dsets'], self.cfg['dropouts'][type][1:])}]
        for type in self.cfg['lrates'].keys():
            self.lrates[type] = {name:lrate for name, lrate in zip(self.cfg['dsets'], self.cfg['lrates'][type])}

        # Check if loss is specified for each dset, otherwise copy it
        assert (len(self.cfg['loss_functions']) == 1 or len(self.cfg['loss_functions']) == len(self.cfg['dsets'])), f"'loss_functions' must be a list with either 1 element, or len(dsets) == {len(self.cfg['dsets'])} elements"
        if len(self.cfg['loss_functions']) != len(self.cfg['dsets']):
            print('Copying loss functions')
            self.cfg['loss_functions'] = [copy.deepcopy(self.cfg['loss_functions'][0]) for _ in self.cfg['dsets']]
        self.loss_functions = {name:fct for name, fct in zip(self.cfg['dsets'], self.cfg['loss_functions'])}

        # Check if optimizer is specified for each dset, otherwise copy it
        assert (len(self.cfg['optimizers']) == 1 or len(self.cfg['optimizers']) == len(self.cfg['dsets'])), f"'optimizers' must be a list with either 1 element, or len(dsets) == {len(self.cfg['dsets'])} elements"
        if len(self.cfg['optimizers']) != len(self.cfg['dsets']):
            print('Copying optimizers')
            self.cfg['optimizers'] = [copy.deepcopy(self.cfg['optimizers'][0]) for _ in self.cfg['dsets']]
        self.optimizers = {name:optim for name, optim in zip(self.cfg['dsets'], self.cfg['optimizers'])}

        # Give an image size for each dset
        # If not specified, pass None to avoid resizing
        if self.cfg.get('working_sizes', None) is None:
            self.cfg['working_sizes'] = [None for _ in range(len(self.cfg['dsets']))]
        # If only 1 value is specified, consider it to be for each dset
        if len(self.cfg['working_sizes']) == 1 and len(self.cfg['dsets']) > 1:
            self.cfg['working_sizes'] = [self.cfg['working_sizes'][0] for _ in range(len(self.cfg['dsets']))]
        # Convert to dict with dset name as key
        self.img_dims = {name:sizes for name, sizes in zip(self.cfg['dsets'], self.cfg['working_sizes'])}

        # Set mu_var_loss as False if not specified (since it is only for autoencoding)
        if self.cfg.get('mu_var_loss', None) is None:
            self.cfg['mu_var_loss'] = False

        self.train_args = {
            'paths' : self.paths,
            'dsets_names' : self.cfg['dsets'],
            'architecture' : self.cfg['architecture'],
            'initialization' : self.cfg['initialization'],
            'bootstrap_size' : max(self.cfg['bootstrap_size'], 1),          # Ensure at least 1 for initialization
            'bootstrap_images': self.cfg.get('bootstrap_images', 1e10),
            'nb_batches_per_epoch' : self.cfg['nb_batches_per_epoch'],
            'nb_epochs' : self.cfg['epochs'],
            'lrates' : self.lrates,
            'dropouts' : self.dropouts,
            'batch_size' : self.cfg['batch_size'],
            'loss_functions' : self.loss_functions,
            'optim_class' : self.optimizers,
            'opt_metrics' : self.cfg['opt_metrics'],
            'metrics_freq': self.cfg['metrics_freq'],
            'training_task': self.cfg['training_task'],
            'mu_var_loss': self.cfg['mu_var_loss'],
            'cuda_device' : self.cuda_device,
            'resume' : self.cfg['resume'],
            'img_dims' : self.img_dims,
            'remove_mode': self.cfg['remove_mode']
        }

        print(f"> Config loaded successfully for {self.cfg['transfer']} training:")
        # Print the config so it is written in the log file as well
        for key, value in self.train_args.items():
            if key == 'paths': continue
            print(f">> {key}: {value}")


    def make_paths(self):
        # Local level/debug config: shorter runs
        # Simply add your `os.uname().nodename` to the list.
        is_local_run = os.uname().nodename in ['example_os_name']
        if is_local_run:
            self.cfg.update({
                'nb_batches_per_epoch' : 1,
                'epochs' : {
                    'train' : 10,
                    'spec' : 10,
                },
                # Paths and filenames
                'dsets_path' : 'datasets/tiny',
                'bootstrap_size' : 10,
            })

        # Make Path objects
        self.cfg.update({
            'dsets_path' : Path(self.cfg['dsets_path']),
            'ramdir' : Path(self.cfg['ramdir']),
            'out_path' : Path(self.cfg['out_path']),
            'exp_file' : Path(self.cfg['exp_file']),
        })

        # Copy dataset to ram for optimization
        # The slash operator '/' in the pathlib module is similar to os.path.join()
        dsets_path_ram = self.cfg['ramdir'] / self.cfg['dsets_path']
        if not is_local_run and not dsets_path_ram.is_dir():
            import shutil
            shutil.copytree(self.cfg['dsets_path'], dsets_path_ram)

        # All paths in one place
        if self.cfg['timestamp']:
            # Add timestamp in folder name to avoid duplicates
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        else:
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}"

        assert (not self.cfg['resume']) or experiment_path.is_dir(), ("Folder experiment does not exist, "
            "either run experiment from the beginning or remove timestamp from folder name")

        models_path = experiment_path / 'models'
        self.paths = {
            'experiment' : experiment_path,
            # Brutelogger logs
            'logs' : experiment_path / 'run_logs',
            'dsets' : {d: self.cfg['dsets_path'] / f"{d}" for d in self.cfg['dsets']}
                            if is_local_run else {d: dsets_path_ram / f"{d}" for d in self.cfg['dsets']},
            # Trained model (no specialization)
            # p for parallel and s for sequential
            'train_model_p' : models_path / 'train_model_p.pth',
            'train_model_s' : models_path / 'train_model_s.pth',
            # Model saved after the "normal training" in hydra
            'gen_model_s' : models_path / 'gen_model_s.pth',
            # Specialized models
            'spec_models_p' : {d: models_path / f"spec_model_{d}_p.pth" for d in self.cfg['dsets']},
            'spec_models_s' : {d: models_path / f"spec_model_{d}_s.pth" for d in self.cfg['dsets']},
            # bootstrap model
            'bootstrap_model' : models_path / 'bootstrap_model.pth',
            # Plots
            'metrics' : experiment_path / 'run_plot',
            'samples_training' : experiment_path / 'run_plot' / 'samples_training',
            'samples_spec' : experiment_path / 'run_plot' / 'samples_spec'
        }

        # Create directories
        self.paths['metrics'].mkdir(parents=True, exist_ok=True)
        self.paths['samples_training'].mkdir(parents=True, exist_ok=True)
        self.paths['samples_spec'].mkdir(parents=True, exist_ok=True)
        self.paths['logs'].mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)


    def main_run(self):
        start = time.perf_counter()
        # Need this for sequential learning
        assert self.cfg['trg_dset'] == self.cfg['dsets'][0], "Target dataset must be in first position"
        assert self.cfg['transfer'] in ['sequential', 'parallel'], "Please transfer argument must be 'sequential' or 'parallel'"
        # Copy the experiment.py and exp cfg file in the experiment dir
        shutil.copy2(self.cfg['exp_file'], self.paths['experiment'])
        shutil.copy2('experiment.py', self.paths['experiment'])

        self.typhon = typhon.Typhon(**self.train_args)
        # Bootstrap initialization
        if self.cfg['initialization'] == 'bootstrap':
            # Remove old bootstrap file
            if self.paths['bootstrap_model'].is_file():
                print("> Removing old bootstrap model:", self.paths['bootstrap_model'])
                os.remove(self.paths['bootstrap_model'])
            # Initialize new bootstrap model
            self.typhon.bootstrap()
        # Random initialization
        if self.cfg['initialization'] == 'random':
            # Remove old bootstrap file
            if self.paths['bootstrap_model'].is_file():
                print("> Removing old bootstrap model:", self.paths['bootstrap_model'])
                os.remove(self.paths['bootstrap_model'])
            # Initialize new random model
            self.typhon.random_initialization()
        # Security check
        if  self.cfg['initialization'] == 'load':
            assert self.paths['bootstrap_model'].is_file(), f"Bootstrap initialization file missing ({self.paths['bootstrap_model']}), please choose another initialization"
            print("> Loading Bootstrap initialization from ", self.paths['bootstrap_model'])
        if self.cfg['transfer'] == 'sequential':
            self.typhon.s_train(self.paths['bootstrap_model'])
            if self.cfg['epochs']['spec'] > 0: self.typhon.s_specialization(self.paths['train_model_s'])
        if self.cfg['transfer'] == 'parallel':
            if self.cfg['resume']:
                self.typhon.p_train(self.paths['train_model_p'])
            else:
                self.typhon.p_train(self.paths['bootstrap_model'])
            if self.cfg['epochs']['spec'] > 0: self.typhon.p_specialization(self.paths['train_model_p'])

        stop = time.perf_counter()
        total_time = stop - start
        print(f"Experiment ended in {int(total_time / 3600)} hours {int((total_time % 3600) / 60)} minutes {total_time % 60:.1f} seconds")
        utils.print_time('END EXPERIMENT')
