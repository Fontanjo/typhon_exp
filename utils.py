import numpy as np
import datetime
import sklearn.metrics
import torch
import torchvision
import glob
import cv2


class SegmentationDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, cuda_device='cpu', path="datasets_segmentation/Brain/test/", img_dim=(256, 256)):
        self.loader = loader
        self.cuda_device = cuda_device
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*[!_mask].npy") # All non-mask
        self.data = []
        for img in file_list:
            mask = img[:-4] + "_mask.npy"
            self.data.append([img, mask])
        self.img_dim = img_dim
        self.num_samples = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_mask_path = self.data[idx]
        img = self.loader(img_path)
        img_mask = self.loader(img_mask_path, True)

        return img, img_mask



class LoopLoader():
    def __init__(self,
            dset_path,
            which, # '['train', 'test', 'val']'
            batch_size,
            cuda_device,
            training_task='classification'
        ):

        self.dset_path = dset_path
        self.which = which
        self.batch_size = batch_size
        self.cuda_device = cuda_device
        self.training_task = training_task

        if self.training_task == 'classification':
            # For a list of which, we concatenate
            self.ds_folder = torch.utils.data.ConcatDataset([torchvision.datasets.DatasetFolder(
                root=f"{dset_path}/{split}",
                extensions="npy",
                loader=gen_loader(self.cuda_device))
                for split in which])
        elif self.training_task == 'segmentation':
            self.ds_folder = torch.utils.data.ConcatDataset([SegmentationDatasetFolder(
            path=f"{dset_path}/{split}/",
            cuda_device='cpu',
            loader=segmentation_loader(self.cuda_device))
            for split in which])
        else:
            # Todo trow errow and say that task is not valid but shoul be in ['classification', 'segmentation']
            pass

        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)

        self.reload_iter()


    def reload_iter(self):
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)
        self.loader_iter = iter(self.data_loader)


    def get_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.reload_iter()
            return next(self.loader_iter)


def print_time(str):
    print(str, '--', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def segmentation_loader(cuda_device):
    def the_loader(path, mask=False):
        # Load data
        ary = np.load(path)
        # If mask, add third channel
        if mask: ary.shape = (1, *ary.shape)
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(ary).to(cuda_device)
        return tensor.float()
    return the_loader


# Just putting the cuda_device in a closure for the DatasetFolder loader
def gen_loader(cuda_device):
    # Load an image, convert it to a tensor with one single channel,
    # and send it to the cuda device (GPU/CPU)
    def the_loader(path):
        # Load the data: a 2D numpy array
        ary = np.load(path)
        # We need a 3rd dimension for "channels"
        ary.shape = (1, *ary.shape) # same as `reshape()` but "inplace"
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(ary).to(cuda_device)
        return tensor
    return the_loader


# Returns a big dictionary with all the metrics
def get_metrics(loss_function, confusion_matrix_dict, predictions_per_batch):
    # Get totals
    tp = sum(confusion_matrix_dict['TP'])
    fp = sum(confusion_matrix_dict['FP'])
    tn = sum(confusion_matrix_dict['TN'])
    fn = sum(confusion_matrix_dict['FN'])

    if tp + fp + tn + fn:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    else:
        accuracy = 0.0

    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall:
        f1score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1score = 0.0

    if tn + fp:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    y_true = predictions_per_batch['labels']
    y_score = predictions_per_batch['predictions_positive_class']
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)

    # we use torch.nn.CrossEntropyLoss (`cel` below)
    cel_input = predictions_per_batch['raw_predictions']
    cel_target = predictions_per_batch['labels_tensor']
    loss = loss_function(cel_input, cel_target).item()

    return {
        'loss': loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1score': f1score, 'specificity': specificity, 'auc': auc}


def get_segmentation_metrics(loss_function, predictions_per_batch):
    cel_input = predictions_per_batch['raw_predictions']
    cel_target = predictions_per_batch['labels_tensor']
    loss = loss_function(cel_input, cel_target).item()
    return {
        'loss': loss, 'accuracy': 0,
        'precision': 0, 'recall': 0,
        'f1score': 0, 'specificity': 0, 'auc': 0}
