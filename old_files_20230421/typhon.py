import torch
import time
import copy
import pickle
from tqdm import tqdm
import sklearn.metrics
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from typhon_model import TyphonModel
import utils


class Typhon(object):
    def __init__(self,
            paths,
            dsets_names,
            architecture,
            initialization,
            bootstrap_size,
            bootstrap_images,
            nb_batches_per_epoch,
            nb_epochs,
            lrates,
            dropouts,
            loss_functions,
            optim_class,
            opt_metrics,
            metrics_freq,
            training_task,
            mu_var_loss,
            batch_size,
            cuda_device,
            resume,
            img_dims,
            remove_mode
        ):

        self.paths = paths
        self.dsets_names = dsets_names
        self.architecture = architecture
        self.initialization = initialization
        self.bootstrap_size = bootstrap_size
        self.bootstrap_images = bootstrap_images
        self.nb_batches_per_epoch = nb_batches_per_epoch
        self.nb_epochs = nb_epochs
        self.lrates = lrates
        self.dropouts = dropouts
        self.loss_functions = loss_functions
        self.optim_class = optim_class
        self.opt_metrics = opt_metrics
        self.metrics_freq = metrics_freq
        self.training_task = training_task
        self.mu_var_loss = mu_var_loss
        self.batch_size = batch_size
        self.cuda_device = cuda_device
        self.resume = resume
        self.metrics_plot = pd.DataFrame(columns=['type', 'feature_extractor', 'epoch', 'dataset', 'split', 'metric', 'value'])
        self.best_models = {}
        self.nb_dataset = len(self.paths['dsets'])
        self.img_dims = img_dims
        self.remove_mode = remove_mode
        # assert self.nb_dataset == 3, 'Double check as long as we work with 3'


    @torch.no_grad()
    def test_model(self, model, dset_name, test_data_loader, verbose=False):
        if self.training_task == 'segmentation':
            return self.test_segmentation_model(model, dset_name, test_data_loader, verbose)
        if self.training_task == 'autoencoding':
            return self.test_autoencoding_model(model, dset_name, test_data_loader, verbose)
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        # Send model to GPU if available
        model.to(self.cuda_device)

        # List of predictions to compute AUC (float)
        predictions_per_batch = {'labels': [], 'predictions_positive_class': [], 'raw_predictions': torch.tensor([]).to(self.cuda_device), 'labels_tensor': torch.tensor([]).to(self.cuda_device)}

        confusion_matrix_dict = {}

        start = time.perf_counter()

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with CrossEntropyLoss)
            outputs = model(inputs, dset_name)
            _, predicted = torch.max(outputs, 1)

            # Probabilities required to compute roc_auc_score, so use a softmax
            softmax = torch.nn.Softmax(dim=1)
            proba_classes = softmax(outputs)
            all_positives = torch.index_select(outputs, 1, torch.tensor([1]).to(self.cuda_device))

            predictions_per_batch['labels'] = predictions_per_batch['labels'] + labels.cpu().numpy().tolist()
            predictions_per_batch['predictions_positive_class'] = predictions_per_batch['predictions_positive_class'] + all_positives.cpu().numpy().flatten().tolist()
            predictions_per_batch['raw_predictions'] = torch.cat((predictions_per_batch['raw_predictions'], outputs), 0)
            predictions_per_batch['labels_tensor'] = torch.cat((predictions_per_batch['labels_tensor'].long(), labels), 0)

            (tn, fp), (fn, tp) = sklearn.metrics.confusion_matrix(
                labels.cpu(), predicted.cpu(), labels=[0,1])
            conf_matrix_per_batch = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

            # Tranform dictionary of {str: int} into {str: list(int)} => required for utils.get_metrics()
            # conf_matrix_per_batch = {key: [value] for (key, value) in conf_matrix_per_batch.items()}
            for key, value in conf_matrix_per_batch.items():
                confusion_matrix_dict.setdefault(key, []).append(value)

        end = time.perf_counter()

        metrics_test =  utils.get_metrics(self.loss_functions[dset_name], confusion_matrix_dict, predictions_per_batch)

        if verbose:
            summary_text = f"""
            SUMMARY OF THE CLASSIFIER ON TEST SET :

            -------------------
            Loss: {metrics_test['loss']}
            Accuracy: {metrics_test['accuracy']}
            Precision:{metrics_test['precision']}
            Recall:   {metrics_test['recall']}
            F1 score: {metrics_test['f1score']}
            Specificity: {metrics_test['specificity']}
            AUC: {metrics_test['auc']}
            --------------------
            Running time: {end-start}

            """

            print(summary_text)

        return metrics_test

    @torch.no_grad()
    def test_segmentation_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        # Send model to GPU if available
        model.to(self.cuda_device)

        # List of predictions)
        # predictions_per_batch = {'labels': [], 'predictions_positive_class': [], 'raw_predictions': torch.tensor([]).to(self.cuda_device), 'labels_tensor': torch.tensor([]).to(self.cuda_device)}

        # List of losses
        losses = []

        # List of auc
        aucs = []

        confusion_matrix_dict = {}

        start = time.perf_counter()

        # For each batch
        for inputs, labels in test_data_loader:
            # Reinitialize
            raw_predictions = torch.tensor([]).to(self.cuda_device)
            labels_tensor = torch.tensor([]).to(self.cuda_device)

            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with CrossEntropyLoss)
            outputs = model(inputs, dset_name)


            #####################################################
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # Storing all this is what causes the CUDA problem! #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            #####################################################
            # predictions_per_batch['raw_predictions'] = torch.cat((predictions_per_batch['raw_predictions'], outputs), 0)
            # predictions_per_batch['labels_tensor'] = torch.cat((predictions_per_batch['labels_tensor'].long(), labels), 0)
            # Instead, compute directly the loss (add a group dimension as if there where multiple batches)


            # TODO: improve, useless to create empty tensor and then concatenate only 1 argument
            raw_predictions = torch.cat((raw_predictions, outputs), 0)
            labels_tensor = torch.cat((labels_tensor, labels), 0)

            # Compute loss
            ls = self.loss_functions[dset_name](raw_predictions, labels_tensor).item()
            losses.append(ls)

            # Compute auc
            labels_list = labels.flatten().cpu().numpy().tolist()
            positive_pred_list = outputs.flatten().cpu().numpy().tolist()

            # Ensure labels contains only 0s and/or 1s
            # assert set(labels_list) <= set([0,1]), f'To compure AUC, labels must contain only 0s and 1s'
            try:
                auc = sklearn.metrics.roc_auc_score(labels_list, positive_pred_list).item()
            except ValueError as e:
                auc = 0
            aucs.append(auc)

            # Set the values of the output to 0 or 1 (tumor at pixel xy or not) and cast to int
            predicted = (outputs > 0.5).int()
            labels = labels.int()

            # TODO check efficiency and possibly find a better way
            tp = torch.sum((predicted==labels) * (predicted==1)).item()
            tn = torch.sum((predicted==labels) * (predicted==0)).item()
            fp = torch.sum((predicted!=labels) * (predicted==1)).item()
            fn = torch.sum((predicted!=labels) * (predicted==0)).item()

            conf_matrix_per_batch = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

            for key, value in conf_matrix_per_batch.items():
                confusion_matrix_dict.setdefault(key, []).append(value)

        end = time.perf_counter()



        metrics_test = utils.get_segmentation_metrics(losses, aucs, confusion_matrix_dict)
        # metrics_test = utils.get_segmentation_metrics(self.loss_functions[dset_name], predictions_per_batch, confusion_matrix_dict)

        if verbose:
            summary_text = f"""
            SUMMARY OF THE CLASSIFIER ON TEST SET :

            -------------------
            Loss: {metrics_test['loss']}
            Accuracy: {metrics_test['accuracy']}
            Precision:{metrics_test['precision']}
            Recall:   {metrics_test['recall']}
            F1 score: {metrics_test['f1score']}
            Specificity: {metrics_test['specificity']}
            IoU: {metrics_test['iou']}
            --------------------
            Running time: {end-start}

            """

            print(summary_text)

        return metrics_test


    @torch.no_grad()
    def test_autoencoding_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        # Send model to GPU if available
        model.to(self.cuda_device)

        # List of predictions)
        # predictions_per_batch = {'labels': [], 'predictions_positive_class': [], 'raw_predictions': torch.tensor([]).to(self.cuda_device), 'labels_tensor': torch.tensor([]).to(self.cuda_device)}

        # List of losses
        losses = []

        # List of auc
        aucs = []

        confusion_matrix_dict = {}

        start = time.perf_counter()

        # For each batch
        for inputs, labels in test_data_loader:
            # Reinitialize
            raw_predictions = torch.tensor([]).to(self.cuda_device)
            labels_tensor = torch.tensor([]).to(self.cuda_device)

            input, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)

            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with CrossEntropyLoss)
            outputs = model(inputs, dset_name)

            # Some models (in particular VAEs) returns mu and var together with the output, to compute the loss
            if self.mu_var_loss:
                # Extract mu and var
                outputs, mu, var = outputs


            # TODO: improve, useless to create empty tensor and then concatenate only 1 argument
            raw_predictions = torch.cat((raw_predictions, outputs), 0)
            labels_tensor = torch.cat((labels_tensor, labels), 0)

            # Compute loss
            if self.mu_var_loss:
                ls =  self.loss_functions[dset_name](raw_predictions, labels_tensor, mu, var).item()
            else:
                ls = self.loss_functions[dset_name](raw_predictions, labels_tensor).item()
            losses.append(ls)



        end = time.perf_counter()



        metrics_test = utils.get_autoencoding_metrics(losses, aucs, confusion_matrix_dict)
        # metrics_test = utils.get_segmentation_metrics(self.loss_functions[dset_name], predictions_per_batch, confusion_matrix_dict)

        if verbose:
            summary_text = f"""
            SUMMARY OF THE CLASSIFIER ON TEST SET :

            -------------------
            Loss: {metrics_test['loss']}
            Accuracy: {metrics_test['accuracy']}
            Precision:{metrics_test['precision']}
            Recall:   {metrics_test['recall']}
            F1 score: {metrics_test['f1score']}
            Specificity: {metrics_test['specificity']}
            IoU: {metrics_test['iou']}
            --------------------
            Running time: {end-start}

            """

            print(summary_text)

        return metrics_test

    # Load the model from the given model, and set the optimizers
    # type is either 'train' or 'spec'
    def load_model_and_optims(self, model_path, type, frozen=False):
        print(f"> Loading models from {model_path} and optimizers")
        loaded_state_dicts = torch.load(model_path, map_location=self.cuda_device)
        self.dsets_names = loaded_state_dicts['variables']['dsets_names']
        self.model = TyphonModel.from_state_dict(loaded_state_dicts)
        self.model.set_dropout(*self.dropouts[type])
        self.optimizers = {}
        # Send model to GPU if available
        self.model.to(self.cuda_device)
        # Split the model, to be used in specialization
        self.spec_models = self.model.split_typhon()
        for dset_name in self.dsets_names:
            # Send models to GPU if available
            self.spec_models[dset_name].to(self.cuda_device)

        if type == 'train':
            # Additional option for hydra
            if frozen: type = 'frozen'
            for dset_name in self.dsets_names:
                # Here we keep only the parameters of the FE and a specific DM, for each dataset
                params = torch.nn.ParameterList([param for name, param in self.model.named_parameters() if ('fe' in name) or (dset_name in name)])
                optim = self.optim_class[dset_name](params, lr=self.lrates[type][dset_name])
                self.optimizers[dset_name] = optim

        if type == 'spec':
            # Additional option for hydra
            if frozen: type = 'frozen'
            for dset_name in self.dsets_names:
                # Here we keep only the parameters of the FE and a specific DM, for each dataset
                params = torch.nn.ParameterList([param for name, param in self.spec_models[dset_name].named_parameters() if ('fe' in name) or (dset_name in name)])
                optim = self.optim_class[dset_name](params, lr=self.lrates[type][dset_name])
                self.optimizers[dset_name] = optim

        print(f"> Models and optimizers loaded")


    # type is either 'train', 'spec' or 'bootstrap'
    def load_data(self, type, classic_loader=False):
        print(f"> Loading data")

        if type == 'bootstrap':
            self.bootstrap_data_loaders = {}

            for dset_name in self.dsets_names:
                bootstrap_loop_loader = utils.LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    # Use both train and val sets, more data for bootstrap!
                    which=['train', 'val'],
                    batch_size=self.batch_size['train'],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task,
                    img_dim=self.img_dims[dset_name],
                    dummy=classic_loader,
                    remove_mode=self.remove_mode)

                self.bootstrap_data_loaders[dset_name] = bootstrap_loop_loader.data_loader
                print(f">> Data loaded for dataset {self.paths['dsets'][dset_name]}")

        # For 'train' or 'spec'
        else:
            self.train_loop_loaders = {}
            self.train_data_loaders = {}
            self.validation_data_loaders = {}
            self.test_data_loaders = {}

            for dset_name in self.dsets_names:
                train_loop_loader = utils.LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['train'],
                    batch_size=self.batch_size[type],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task,
                    img_dim=self.img_dims[dset_name],
                    dummy=classic_loader,
                    remove_mode=self.remove_mode)

                validation_loop_loader = utils.LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['val'],
                    batch_size=self.batch_size[type],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task,
                    img_dim=self.img_dims[dset_name],
                    dummy=classic_loader,
                    remove_mode=self.remove_mode)

                test_loop_loader = utils.LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['test'],
                    batch_size=1,
                    cuda_device=self.cuda_device,
                    training_task=self.training_task,
                    img_dim=self.img_dims[dset_name],
                    dummy=classic_loader,
                    remove_mode=self.remove_mode)

                self.train_loop_loaders[dset_name] = train_loop_loader
                self.train_data_loaders[dset_name] = train_loop_loader.data_loader
                self.validation_data_loaders[dset_name] = validation_loop_loader.data_loader
                self.test_data_loaders[dset_name] = test_loop_loader.data_loader

                print(f""">> Data loaded for dataset {self.paths['dsets'][dset_name]}
                    train: {len(train_loop_loader.ds_folder)} images
                    validation: {len(validation_loop_loader.ds_folder)} images
                    test: {len(test_loop_loader.ds_folder)} images
                """)
        print(f"> All data loaded")


    # Train one model on one batch from one dataset
    def train_on_batch(self, model, dset_name, batch):
        assert model.training == True, "Model not in training mode"
        inputs, labels = batch
        # if self.training_task == 'autoencoding':
        #     mode = inputs[0][:, :210, :160].mode()[0].mode()[0]
        #     mode_tensor = torch.tensor([mode[0], mode[1], mode[2]]).expand(256, 256, 3).transpose(1, 2).transpose(0, 1).to(self.cuda_device)
        #     inputs[:, :, :210, :160] -= mode_tensor[:, :210, :160]
        #     inputs = torch.abs(inputs)
        #     labels = inputs.detach().clone()
        # Clear old gradient (default is to accumulate)
        self.optimizers[dset_name].zero_grad()
        # Send data to GPU if available
        inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
        # Run the model on the batch and get predictions
        predictions = model(inputs, dset_name)
        # Some models (in particular VAEs) returns mu and var together with the output, to compute the loss
        if self.mu_var_loss:
            # Extract mu and var
            predictions, mu, logvar = predictions
            # Compute loss between prediction and labels
            loss = self.loss_functions[dset_name](predictions, labels, mu, logvar)
        else:
            # Compute loss between prediction and labels
            loss = self.loss_functions[dset_name](predictions, labels)
        # Backpropagation computes dloss/dx for each x param
        loss.backward()
        # Optimizer.step performs a parameter update based on gradients
        self.optimizers[dset_name].step()


    # train_on is either 'all' or 'some' batch(es)
    def train_step(self, model, dset_name, train_on):
        assert train_on in ['all', 'some'], "train_on must be either 'all' or 'some'"
        if train_on == 'some':
            print(f">>> Training on {self.nb_batches_per_epoch} batches")
            for nbatch in range(self.nb_batches_per_epoch):
                batch = self.train_loop_loaders[dset_name].get_batch()
                self.train_on_batch(model, dset_name, batch)

        if train_on == 'all':
            print(f">>> Training on all batches")
            for batch in self.train_data_loaders[dset_name]:
                self.train_on_batch(model, dset_name, batch)


    # Compute metrics
    def compute_metrics(self, model, dset_name):
        print(f">>> Collecting performance on training set")
        metrics_training = self.test_model(
            model=model,
            dset_name=dset_name,
            test_data_loader=self.train_data_loaders[dset_name])

        print(f">>> Collecting performance on validation set")
        metrics_validation = self.test_model(
            model=model,
            dset_name=dset_name,
            test_data_loader=self.validation_data_loaders[dset_name])

        return metrics_training, metrics_validation


    # type is either 'train' or 'spec'
    def compare_models(self, model, dset_name, type, save_path, epoch, metrics_validation):
        # At first epoch save the model and the score to have a baseline
        if epoch == 0 or not getattr(self, "best_metrics_dict", None):
            self.best_metrics_dict = copy.deepcopy(metrics_validation)
            self.best_metrics_dict['epoch'] = epoch
            self.best_models[dset_name] = copy.deepcopy(model)
            torch.save(self.best_models[dset_name].to_state_dict(), save_path)
            print(f">>> First model saved: {self.opt_metrics[type]}: {self.best_metrics_dict[self.opt_metrics[type]]}")
            pass

        # Compare scores and save model if better
        new_opt = metrics_validation[self.opt_metrics[type]]
        best_opt = self.best_metrics_dict[self.opt_metrics[type]]

        if new_opt > best_opt :
            utils.print_time(">>> Saving new best model")
            print(f">>> New best: {self.opt_metrics[type]}: {best_opt} -> {new_opt}")
            # Setting new best data
            self.best_metrics_dict = copy.deepcopy(metrics_validation)
            self.best_metrics_dict['epoch'] = epoch
            # Save model
            self.best_models[dset_name] = copy.deepcopy(model)
            torch.save(self.best_models[dset_name].to_state_dict(), save_path)
            print(f">>> New best model saved at epoch {self.best_metrics_dict['epoch']},", end=' ')
            print(f"{self.opt_metrics[type]}: {self.best_metrics_dict[self.opt_metrics[type]]}")


    # Save a sample of the current model
    def save_sample(self, path, model, dset_name, epoch):
        if epoch != 0: epoch += 1 # Except for first one (before training), save_sample is called checking (epoch+1) % 0
        data_loader = self.test_data_loaders[dset_name]
        # Load 1 batch
        inputs, labels = next(iter(data_loader)) # Access only 1 batch
        # Pass to model
        outputs = model(inputs, dset_name)
        # Some models (in particular VAEs) returns mu and var together with the output, to compute the loss
        if self.mu_var_loss:
            outputs, mu, var = outputs
        # Convert to numpy
        inp, out, lab = inputs.cpu().detach().numpy(), outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        # Select first image of each batch and move color channel at the end
        inp, out, lab = inp[0].transpose(1, 2, 0), out[0].transpose(1, 2, 0), lab[0].transpose(1, 2, 0)


        # cv2.imwrite(str(self.paths['samples'] / f'ep{epoch}_{dset_name}_input.jpg'), inp * 255)
        cv2.imwrite(path + f'/ep{epoch}_{dset_name}_input.png', inp * 255)
        cv2.imwrite(path + f'/ep{epoch}_{dset_name}_output.png', out * 255)
        cv2.imwrite(path + f'/ep{epoch}_{dset_name}_label.png', lab * 255)



###############################################################################################################################
############################ PARALLEL TRANSFER LEARNING #######################################################################
###############################################################################################################################
    def p_train(self, model_path):
        # Typhon has external loop for epochs, then loops on the
        # datasets in turn, and for each it trains on a single batch for each epoch.
        utils.print_time("PARALLEL TRAINING")
        self.load_data('train')
        self.load_model_and_optims(model_path, 'train')
        range_epochs = range(self.nb_epochs['train'])

        if self.resume:
            assert (self.paths['metrics'] / 'metrics.csv').is_file(), "Cannot resume empty experiment"
            # index_col avoids adding new column and take first column as index
            self.metrics_plot = pd.read_csv(self.paths['metrics'] / 'metrics.csv', index_col=0)
            # Delete 'test' results and 'specialized' metrics if any
            self.metrics_plot.drop(self.metrics_plot[self.metrics_plot['split'] == 'test'].index, inplace=True)
            self.metrics_plot.drop(self.metrics_plot[self.metrics_plot['type'] == 'specialized'].index, inplace=True)
            start_epoch = self.metrics_plot['epoch'].max() + 1
            range_epochs = range(start_epoch, start_epoch + self.nb_epochs['train'])
            print(f"> Resuming training from epoch {start_epoch}")

        # Save a first sample, to visualize bootstrap output
        for dset_name in self.dsets_names:
            self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=0)

        for epoch in tqdm(range_epochs):
            print(f">> Epoch {epoch}")

            for dset_name in self.dsets_names:
                print(f">>> Dset {dset_name}")
                self.model.train()
                self.train_step(self.model, dset_name, 'some')
                if (epoch + 1) % self.metrics_freq['train'] == 0:
                    metrics_training, metrics_validation = self.compute_metrics(self.model, dset_name)
                    # Add training and validation metrics for this epoch
                    print(f">>> Aggregating metrics and saving")
                    self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'trained', 'unfrozen')
                    self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'trained', 'unfrozen')
                    print(f">>> {self.opt_metrics['train']} train: {metrics_training[self.opt_metrics['train']]} ")
                    print(f">>> {self.opt_metrics['train']} val: {metrics_validation[self.opt_metrics['train']]} ")
                    # Save a sample
                    self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=epoch)
                # Save after each epoch, so we can quit and resume at any time
                model_state = copy.deepcopy(self.model.to_state_dict()) # to_state_dict returns a reference to the state and not the copy, thus it will be modified
                torch.save(model_state, self.paths['train_model_p'])

        # Test and save trained models
        print(f"> Models training completed, testing now")
        for dset_name in self.dsets_names:
            print(f">> Results for {dset_name}, WITHOUT specialization")
            metrics_test = self.test_model(
                model=self.model,
                dset_name=dset_name,
                test_data_loader=self.test_data_loaders[dset_name],
                verbose=True)

            self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'trained', 'unfrozen')

        model_state = self.model.to_state_dict()
        torch.save(model_state, self.paths['train_model_p'])
        print(f"> Training complete")


    # Specialization after the parallel training
    def p_specialization(self, model_path):
        utils.print_time("SPECIALIZATION")
        self.load_data('spec')
        self.load_model_and_optims(model_path, 'spec')
        # Best model per each epoch to simulate early stopping on max validation
        best_spec_dict = {}
        best_spec_models = {}

        # Save a first sample, to visualize bootstrap output
        for dset_name in self.dsets_names:
            self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=0)

        for dset_name in self.dsets_names:
            utils.print_time(f">> Dataset {dset_name}")

            # Loop for the specialization epochs
            for epoch in tqdm(range(self.nb_epochs['spec'])):
                print(f">>> Epoch {epoch}")
                self.spec_models[dset_name].train()
                self.train_step(self.spec_models[dset_name], dset_name, 'all')
                if (epoch + 1) % self.metrics_freq['train'] == 0:
                    metrics_training, metrics_validation = self.compute_metrics(self.model, dset_name)
                    print(f">>> Aggregating metrics")
                    self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'specialized', 'unfrozen')
                    self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'specialized', 'unfrozen')
                    # Save a sample
                    self.save_sample(path=str(self.paths['samples_spec']), model=self.model, dset_name=dset_name, epoch=epoch)

                    self.compare_models(
                        model=self.spec_models[dset_name],
                        dset_name=dset_name,
                        type='spec',
                        save_path=self.paths['spec_models_p'][dset_name],
                        epoch=epoch,
                        metrics_validation=metrics_validation)

            # Test the best model (the one that has been saved)
            print(f">> Results for {dset_name}, WITH specialization")
            metrics_test = self.test_model(
                model=self.best_models[dset_name],
                dset_name=dset_name,
                test_data_loader=self.test_data_loaders[dset_name],
                verbose=True)

            self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'specialized', 'unfrozen')

        print(f"> Specialization completed")


###############################################################################################################################
############################ SEQUENTIAL TRANSFER LEARNING #####################################################################
###############################################################################################################################
    def s_train(self, model_path):
        # Hydra loops on the datasets in turn
        # and has internal loop for epochs
        utils.print_time("SEQUENTIAL TRAINING")
        self.load_data('train', classic_loader=True)
        self.load_model_and_optims(model_path, 'train')

        # For each dataset, Hydra trains on some epochs for all batches
        for idx, dset_name in enumerate(self.dsets_names):
            print(f">> Dset {dset_name}")

            for feature_extractor in ['frozen', 'unfrozen']:
                # Only train with unfrozen feature extractor for the first dataset
                if idx == 0 and feature_extractor == 'frozen':
                    continue

                # Initialization for further use
                best_train_dict = {}
                best_train_model = {}

                # First passage with frozen FE
                if feature_extractor == 'frozen':
                    self.load_model_and_optims(self.paths['train_model_s'], 'train', frozen=True)
                    print(f">>> Train {dset_name} with frozen feature extractor")

                # Second passage with unfrozen FE
                if feature_extractor == 'unfrozen':
                    if idx != 0:
                        self.load_model_and_optims(self.paths['train_model_s'], 'train', frozen=False)
                    print(f">>> Train {dset_name} with unfrozen feature extractor")

                # Save a first sample, to visualize bootstrap output
                self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=0)
                for epoch in tqdm(range(self.nb_epochs['train'])):
                    print(f">>>> Epoch {epoch}")
                    self.model.train()
                    if feature_extractor == 'frozen': self.model.freeze_fe()
                    if feature_extractor == 'unfrozen': self.model.unfreeze_fe()
                    self.train_step(self.model, dset_name, 'all')
                    if (epoch + 1) % self.metrics_freq['train'] == 0:
                        metrics_training, metrics_validation = self.compute_metrics(self.model, dset_name)
                        # Add training and validation metrics for this epoch
                        print(f">>>> Aggregating metrics")
                        self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'trained', feature_extractor)
                        self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'trained', feature_extractor)
                        print(f">>> {self.opt_metrics['train']} train: {metrics_training[self.opt_metrics['train']]} ")
                        print(f">>> {self.opt_metrics['train']} val: {metrics_validation[self.opt_metrics['train']]} ")
                        # Save a sample
                        self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=epoch)
                        if (feature_extractor == 'unfrozen') and (idx == 0):
                            # Save also the very first base model, after the "normal training"
                            self.compare_models(
                                model=self.model,
                                dset_name=dset_name,
                                type='train',
                                save_path=self.paths['gen_model_s'],
                                epoch=epoch,
                                metrics_validation=metrics_validation)

                        self.compare_models(
                            model=self.model,
                            dset_name=dset_name,
                            type='train',
                            save_path=self.paths['train_model_s'],
                            epoch=epoch,
                            metrics_validation=metrics_validation)

            # Test first (target) dataset
            if idx == 0:
                print(f">> Results for {dset_name}, WITHOUT specialization")
                metrics_test = self.test_model(
                    model=self.best_models[dset_name],
                    dset_name=dset_name,
                    test_data_loader=self.test_data_loaders[dset_name],
                    verbose=True)

                self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'trained', 'unfrozen')

        print(f"> Training complete")


    # Specialization after the sequential training
    def s_specialization(self, model_path):
        utils.print_time("SPECIALIZATION")
        self.load_data('spec')
        self.load_model_and_optims(model_path, 'spec', frozen=True)

        # Specialization only on the first/target dataset
        dset_name = self.dsets_names[0]

        utils.print_time(f">> Dataset {dset_name}")
        # Best model per each epoch to simulate early stopping on max validation
        best_spec_dict = {}
        best_spec_models = {}

        for feature_extractor in ['frozen', 'unfrozen']:
            # First passage with frozen FE
            if feature_extractor == 'frozen':
                print(f">> Train {dset_name} with frozen feature extractor")

            # Second passage with unfrozen FE
            if feature_extractor == 'unfrozen':
                self.load_model_and_optims(self.paths['spec_models_s'][dset_name], 'spec', frozen=False)
                print(f">> Train {dset_name} with unfrozen feature extractor")

            # Save a first sample, to visualize bootstrap output
            self.save_sample(path=str(self.paths['samples_training']), model=self.model, dset_name=dset_name, epoch=0)
            # Loop for the specialization epochs
            for epoch in range(self.nb_epochs['spec']):
                print(f">>> Epoch {epoch}")
                self.spec_models[dset_name].train()
                if feature_extractor == 'frozen': self.spec_models[dset_name].freeze_fe()
                if feature_extractor == 'unfrozen': self.spec_models[dset_name].unfreeze_fe()
                self.train_step(self.spec_models[dset_name], dset_name, 'all')
                if (epoch + 1) % self.metrics_freq['train'] == 0:
                    metrics_training, metrics_validation = self.compute_metrics(self.model, dset_name)
                    print(f">>> Aggregating metrics")
                    self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'specialized', feature_extractor)
                    self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'specialized', feature_extractor)
                    # Save a sample
                    self.save_sample(path=str(self.paths['samples_spec']), model=self.model, dset_name=dset_name, epoch=ep)

                self.compare_models(
                    model=self.spec_models[dset_name],
                    dset_name=dset_name,
                    type='spec',
                    save_path=self.paths['spec_models_s'][dset_name],
                    epoch=epoch,
                    metrics_validation=metrics_validation)

        # Test the best model (the one that has been saved)
        print(f"> Results for {dset_name}, WITH specialization")
        metrics_test = self.test_model(
            model=self.best_models[dset_name],
            dset_name=dset_name,
            test_data_loader=self.test_data_loaders[dset_name],
            verbose=True)

        self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'specialized', 'unfrozen')

        print(f"> Specialization completed")


###############################################################################################################################
############################ BOOTSTRAP ########################################################################################
###############################################################################################################################
    @torch.no_grad()
    def bootstrap(self):
        if self.training_task == 'autoencoding':
            return self.bootstrap_autoencoding()
        utils.print_time("BOOTSTRAP")
        self.load_data('bootstrap')

        best = {dset:{} for dset in self.dsets_names}

        for nmodel in tqdm(range(self.bootstrap_size)):
            # Take the dropouts of the training (no impact since we only test)
            dropout_fe, dropouts_dm = self.dropouts['train']

            model = TyphonModel(
                dropout_fe=dropout_fe,
                dropouts_dm=dropouts_dm,
                architecture=self.architecture,
                dsets_names=self.dsets_names)

            nbetterheads = 0
            # Need to reset the dict at each new model
            current = {dset:{} for dset in self.dsets_names}
            current['model'] = model
            # To speed up bootstrap, go to next iteration when the model is bad
            bad_model = False

            for dset_name in self.dsets_names:
                assert self.paths['dsets'][dset_name].stem == dset_name, "Dataset not corresponding to the path"

                # Test model
                print(f">>> {dset_name}")
                metrics_test = self.test_model(
                    model=model,
                    dset_name=dset_name,
                    test_data_loader=self.bootstrap_data_loaders[dset_name])

                current[dset_name] = metrics_test

                # We need a basis model at the first iteration
                if nmodel == 0:
                    best['model'] = model
                    best[dset_name] = metrics_test
                    print(f">>> First iteration for {dset_name}, {self.opt_metrics['bootstrap']}: {best[dset_name][self.opt_metrics['bootstrap']]}")
                    continue

                new_score = current[dset_name][self.opt_metrics['bootstrap']]
# to create bootstrap plot                # print(new_score)
                best_score = best[dset_name][self.opt_metrics['bootstrap']]
                if new_score > best_score:
                    nbetterheads += 1
                    print(f">>> Current better `{self.opt_metrics['bootstrap']}` for {dset_name}: {new_score}")
                # Make sure this is only when using AUC
                if new_score < 0.5 and (self.opt_metrics['bootstrap'] == 'auc'):
                    bad_model = True
                    # Directly go to the next model
                    break

            # Make sure this is only when using AUC
            # Throw the model to speed up bootstrap and avoid computations
            if bad_model and (self.opt_metrics['bootstrap'] == 'auc'):
                print(f">> One head is <0.5 AUC, throw the model")
                continue

            opt_metrics = [current[dset_name][self.opt_metrics['bootstrap']] for dset_name in self.dsets_names]
            if self.opt_metrics['bootstrap'] == 'iou':
                # At least two better heads and a better total
                    # iou mostly around 0 but sometime up to 7/8
                if (nbetterheads > 1) and sum(opt_metrics) > sum([best[dset_name][self.opt_metrics['bootstrap']] for dset_name in self.dsets_names]):
                    print(f">> New best model")
                    best = current
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")
            else:
                # At least two better heads and max difference of 0.2 -> better model
                if (nbetterheads > 1) and ((max(opt_metrics) - min(opt_metrics)) < 0.2):
                    print(f">> New best model")
                    best = current
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")

        torch.save(best['model'].to_state_dict(), self.paths['bootstrap_model'])

        print("> Bootstrap done, best model is saved:")
        for dset_name in self.dsets_names:
            print(f"> {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")


    def random_initialization(self):
        utils.print_time("RANDOM INITIALIZATION")

        # Take the dropouts of the training (no impact since we only test)
        dropout_fe, dropouts_dm = self.dropouts['train']

        model = TyphonModel(
            dropout_fe=dropout_fe,
            dropouts_dm=dropouts_dm,
            architecture=self.architecture,
            dsets_names=self.dsets_names)

        torch.save(model.to_state_dict(), self.paths['bootstrap_model'])

        print("> Random initialization done, model is saved")


    @torch.no_grad()
    def bootstrap_autoencoding(self):
        utils.print_time("BOOTSTRAP")
        self.load_data('bootstrap') # Load train data (and thus validation)

        best = {dset:{} for dset in self.dsets_names}

        def smaller_data_loader(limit, loader):
            i = 0
            for i, el in enumerate(loader):
                i += 1
                if i >= limit:
                    yield el
                    return
                else:
                    yield el

        # Evaluate 250 images
        nb_epochs = int(self.bootstrap_images / self.batch_size['train'])
        assert nb_epochs > 0, f'bootstrap_images ({self.bootstrap_images}) must be >= of training batch size ({self.batch_size["train"]})!'

        for nmodel in tqdm(range(self.bootstrap_size)):
            # Take the dropouts of the training (no impact since we only test)
            dropout_fe, dropouts_dm = self.dropouts['train']

            model = TyphonModel(
                dropout_fe=dropout_fe,
                dropouts_dm=dropouts_dm,
                architecture=self.architecture,
                dsets_names=self.dsets_names)

            # Need to reset the dict at each new model
            current = {dset:{} for dset in self.dsets_names}
            current['model'] = model

            # TODO: allow to use only some dsets in bootstrap, possibly with weights
            for dset_name in self.dsets_names:
                assert self.paths['dsets'][dset_name].stem == dset_name, "Dataset not corresponding to the path"
                # At least for the moment, use loss
                assert self.opt_metrics['bootstrap'] == 'loss'

                # Test model
                print(f">>> {dset_name}")
                metrics_test = self.test_model(
                    model=model,
                    dset_name=dset_name,
                    test_data_loader=smaller_data_loader(nb_epochs, self.bootstrap_data_loaders[dset_name]))
                    # test_data_loader=self.bootstrap_data_loaders[dset_name])
                    # test_data_loader=self.validation_data_loaders[dset_name]) # Use only validation, to be faster (dsets are already generated with the purpose of having different images)

                current[dset_name] = metrics_test

                # We need a basis model at the first iteration
                if nmodel == 0:
                    best['model'] = model
                    best[dset_name] = metrics_test
                    print(f">>> First iteration for {dset_name}, {self.opt_metrics['bootstrap']}: {best[dset_name][self.opt_metrics['bootstrap']]}")
                    continue

                new_score = current[dset_name][self.opt_metrics['bootstrap']]
                best_score = best[dset_name][self.opt_metrics['bootstrap']]


            opt_metrics = [current[dset_name][self.opt_metrics['bootstrap']] for dset_name in self.dsets_names]
            best_metrics = [best[dset_name][self.opt_metrics['bootstrap']] for dset_name in self.dsets_names]

            if sum(opt_metrics) < sum(best_metrics):
                best = current
                for dset_name in self.dsets_names:
                    print(f">>> ep {nmodel}: New {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")
                print(f'>>> Sum: {sum(opt_metrics)}')

            # For distribution learning purpose, comment otherwise
            # for dset_name in self.dsets_names:
            #     print(f">>> ep {nmodel}: {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")
            # print(f'>>> ep {nmodel} sum: {sum(opt_metrics)}')


        torch.save(best['model'].to_state_dict(), self.paths['bootstrap_model'])

        print("> Bootstrap done, best model is saved:")
        for dset_name in self.dsets_names:
            print(f"> {self.opt_metrics['bootstrap']} score for {dset_name}: {best[dset_name][self.opt_metrics['bootstrap']]}")



    def aggregate_metrics(self, metrics, split, dset_name, epoch, type, feature_extractor):
        # Add all training metrics
        for metric, value in metrics.items():
            # Need to be a dataframe to concatenate
            new_row = pd.DataFrame({
                # Type is either trained or specialized
                'type': type,
                # Feature_extractor is either frozen or unfrozen
                'feature_extractor': feature_extractor,
                'epoch': epoch,
                'dataset': dset_name,
                'split': split,
                'metric': metric,
                'value': value,
            # Need to pass an index to concatenate
            }, index=[0])
            self.metrics_plot = pd.concat([self.metrics_plot, new_row], ignore_index=True)

        self.metrics_plot.to_csv(self.paths['metrics'] / 'metrics.csv')
