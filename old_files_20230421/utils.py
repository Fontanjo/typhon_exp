import numpy as np
import datetime
import sklearn.metrics
import torch
import torchvision
import glob
import time


class SegmentationDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, path, cuda_device='cpu', img_dim=None):
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
        # assert img.max() <= 1, f'Wrong format for {self.data[idx]}'
        # assert img.min() >= 0, f'Wrong format for {self.data[idx]}'
        img_mask = self.loader(img_mask_path)
        # Ensure the sizes are correct
        if self.img_dim is not None:
            # Second argument in torch padding requires the size to add "before last dimension", "after last dimension", "before second-to-last dimension", ...
            img = torch.nn.functional.pad(img, (0, self.img_dim[1], 0, self.img_dim[0]), 'constant', 0)[:, :self.img_dim[0], :self.img_dim[1]]
            img_mask = torch.nn.functional.pad(img_mask, (0, self.img_dim[1], 0, self.img_dim[0]), 'constant', 0)[:, :self.img_dim[0], :self.img_dim[1]]

        return img, img_mask


class AutoencodingDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, path, cuda_device='cpu', img_dim=None, remove_mode=True):
        self.loader = loader
        self.cuda_device = cuda_device
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*.npy") # All .npy files
        self.data = []
        for img in file_list:
            self.data.append(img)
        self.img_dim = img_dim
        self.num_samples = len(self.data)
        self.remove_mode = remove_mode
        if self.remove_mode:
            self.mode_img = self.get_dset_mode(100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.loader(img_path)
        # Ensure the sizes are correct
        if self.img_dim is not None:
            # Second argument in torch padding requires the size to add "before last dimension", "after last dimension", "before second-to-last dimension", ...
            img = torch.nn.functional.pad(img, (0, self.img_dim[1], 0, self.img_dim[0]), 'constant', 0)[:, :self.img_dim[0], :self.img_dim[1]]
        if self.remove_mode:
            # Remove mode image
            img -= self.mode_img
            img = torch.abs(img)
        # Return img as input and as label
        return img, img


    def get_dset_mode(self, nb_samples=10):
        imgs = []
        for img_idx in range(min(self.num_samples, nb_samples)):
            img_path = self.data[img_idx]
            img = self.loader(img_path)
            if self.img_dim is not None:
                # Second argument in torch padding requires the size to add "before last dimension", "after last dimension", "before second-to-last dimension", ...
                img = torch.nn.functional.pad(img, (0, self.img_dim[1], 0, self.img_dim[0]), 'constant', 0)[:, :self.img_dim[0], :self.img_dim[1]]
            imgs.append(img.cpu().detach())
        # Merge in a single 4D tensor
        imgs = torch.stack(imgs)
        # Compute and return the mode image
        start = time.time()
        mode_img = imgs.mode(dim=0)[0]
        stop = time.time()
        assert mode_img.shape == (3, 210, 160)
        # Send mode img to same cuda_device as other images
        mode_img = mode_img.to(self.cuda_device)
        print(f'Mode image created (in {round(stop - start, 4)}s)')
        return mode_img



class LoopLoader():
    def __init__(self,
            dset_path,
            which, # '['train', 'test', 'val']'
            batch_size,
            cuda_device,
            training_task='classification',
            img_dim=(256,256),
            dummy=False, # If we need a stardard loader, this deactivate the reshuffling
            remove_mode=True
        ):

        self.dset_path = dset_path
        self.which = which
        self.batch_size = batch_size
        self.cuda_device = cuda_device
        self.training_task = training_task
        self.img_dim = img_dim
        self.dummy = dummy
        self.remove_mode = remove_mode

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
                cuda_device=self.cuda_device,
                loader=segmentation_loader(self.cuda_device),
                img_dim=self.img_dim)
                for split in which])
        elif self.training_task == 'autoencoding':
            self.ds_folder = torch.utils.data.ConcatDataset([AutoencodingDatasetFolder(
                path=f"{dset_path}/{split}/",
                cuda_device=self.cuda_device,
                loader=autoencoding_loader(self.cuda_device),
                img_dim=self.img_dim,
                remove_mode=self.remove_mode)
                for split in which])
        else:
            raise UnrecognizedTaskError("Task is not valid, should be in ['classification', 'segmentation', 'autoencoding']")


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
        if self.dummy: return next(self.loader_iter)
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.reload_iter()
            return next(self.loader_iter)


def print_time(str):
    print(str, '--', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def segmentation_loader(cuda_device):
    def the_loader(path):
        # Load data
        ary = np.load(path)
        # Add color channel if necessary
        assert len(ary.shape) == 2 or len(ary.shape) == 3, f'Expect inputs to have 2 or 3 channels, got {len(ary.shape)}'
        if len(ary.shape) == 2:
            ary.shape = (1, *ary.shape)
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(ary).float().to(cuda_device)
        return tensor
    return the_loader


def autoencoding_loader(cuda_device):
    def the_loader(path):
        # Load data
        ary = np.load(path)
        # print(ary.shape)
        # Move color channel in front
        ary = ary.transpose(2, 0, 1)
        # print(ary.shape)
        # Normalize
        ary = np.divide(ary, 255)
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(ary).float().to(cuda_device)
        return tensor
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


def get_segmentation_metrics(losses, aucs, confusion_matrix_dict):
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

    if tp + fp + fn:
        iou = tp / (tp + fp + fn)
    else:
        iou = 0.0

    # Receive directly the per-batch losses and average them
    loss = np.mean(losses)

    # Receive the per-batch auc and average them
    #  In some cases, the auc can not be computed and is saved as 0 (when only 1 label)
    #  Ignore this cases for the average
    auc = np.mean([a for a in aucs if a != 0])

    return {
        'loss': loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1score': f1score, 'specificity': specificity, 'auc': auc, 'iou': iou}



def get_autoencoding_metrics(losses, aucs, confusion_matrix_dict):
    # Get totals
    # tp = sum(confusion_matrix_dict['TP'])
    # fp = sum(confusion_matrix_dict['FP'])
    # tn = sum(confusion_matrix_dict['TN'])
    # fn = sum(confusion_matrix_dict['FN'])
    #
    # if tp + fp + tn + fn:
    #     accuracy = (tp + tn) / (tp + fp + tn + fn)
    # else:
    #     accuracy = 0.0
    #
    # if tp + fp:
    #     precision = tp / (tp + fp)
    # else:
    #     precision = 0.0
    #
    # if tp + fn:
    #     recall = tp / (tp + fn)
    # else:
    #     recall = 0.0
    #
    # if precision + recall:
    #     f1score = 2 * ((precision * recall) / (precision + recall))
    # else:
    #     f1score = 0.0
    #
    # if tn + fp:
    #     specificity = tn / (tn + fp)
    # else:
    #     specificity = 0.0
    #
    # if tp + fp + fn:
    #     iou = tp / (tp + fp + fn)
    # else:
    #     iou = 0.0

    # Receive directly the per-batch losses and average them
    loss = np.mean(losses)

    # Receive the per-batch auc and average them
    #  In some cases, the auc can not be computed and is saved as 0 (when only 1 label)
    #  Ignore this cases for the average
    # auc = np.mean([a for a in aucs if a is not 0])


    accuracy = 0
    precision = 0
    recall = 0
    f1score = 0
    specificity = 0
    auc = 0
    iou = 0

    return {
        'loss': loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1score': f1score, 'specificity': specificity, 'auc': auc, 'iou': iou}


#############################################
# https://github.com/wolny/pytorch-3dunet/blob/eafaa5f830eebfb6dbc4e570d1a4c6b6e25f2a1e/pytorch3dunet/unet3d/losses.py
#############################################



def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    # print(f'{input.size()}, {target.size()}')
    assert input.size() == target.size(), f"'input' and 'target' must have the same shape, got {input.size()} and {target.size()}"

    input = input.flatten()
    target = target.flatten()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(torch.nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = torch.nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = torch.nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # input = flatten(input)
        input = input.flatten()
        # target = flatten(target)
        target = target.flatten()
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())



class VAELossMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.bce = torch.nn.BCELoss(reduction='sum')
        self.mse = torch.nn.MSELoss()

    def forward(self, raw_predictions, labels_tensor, mu, logvar):
        # Compute MSE loss
        mse_loss = self.mse(raw_predictions, labels_tensor)
        # Compute KL-Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Aggregate them (sum)
        return mse_loss + KLD


class VAELossBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss(reduction='sum')

    def forward(self, raw_predictions, labels_tensor, mu, logvar):
        # Compute BCE loss
        bce_loss = self.bce(raw_predictions, labels_tensor)
        # Compute KL-Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Aggregate them (sum)
        return bce_loss + KLD


class VAELossBCE_MSE(torch.nn.Module):
    def __init__(self, weights=[1, 1, 1]):
        super().__init__()
        self.bce = torch.nn.BCELoss(reduction='sum')
        self.mse = torch.nn.MSELoss()
        self.weights = weights

    def forward(self, raw_predictions, labels_tensor, mu, logvar):
        # Compute BCE loss
        bce_loss = self.bce(raw_predictions, labels_tensor)
        # Compute MSE loss
        mse_loss = self.mse(raw_predictions, labels_tensor)
        # Compute KL-Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Aggregate them (sum)
        return self.weights[0] * bce_loss + self.weights[1] * mse_loss + self.weights[2] * KLD


# Count the numer of pixel that diffears (using torch.isclose)
# default values for relative tolerance (rtol) and absolute tolerance (atol) are those default in the function
#  -> Not differentiable!
# Ignore small differences
class EqualPixels(torch.nn.Module):
    # def __init__(self, rtol=1e-05, atol=1e-08):
    def __init__(self, lambd=0.2):
        super().__init__()
        self.lambd = lambd
        self.remove_small = torch.nn.Hardshrink(lambd=self.lambd)
        # self.rtol = rtol
        # self.atol = atol

    def forward(self, raw_predictions, labels_tensor):
        diff = torch.subtract(raw_predictions, labels_tensor)
        # Remove small values
        hsr_diff = self.remove_small(diff)
        # Get all positive values, with abs
        pos_diff = torch.abs(hsr_diff)
        # Sum
        return torch.sum(pos_diff)
        # # Get True/False for each pixel
        # equal_pixels = torch.isclose(raw_predictions, labels_tensor, rtol=self.rtol, atol=self.atol)
        # # equal_pixels = torch.eq(raw_predictions, labels_tensor)
        # # Count true values (sum)
        # sm = torch.sum(equal_pixels)
        # return sm


class EqualPixels1(torch.nn.Module):
    # def __init__(self, rtol=1e-05, atol=1e-08):
    def __init__(self, lambd=0.2):
        super().__init__()
        self.lambd = lambd
        self.remove_small = torch.nn.Hardshrink(lambd=self.lambd)


    def forward(self, raw_predictions, labels_tensor):
        diff = torch.subtract(raw_predictions, labels_tensor)
        # Remove small values
        hsr_diff = self.remove_small(diff)
        # Get all positive values, by squaring
        pos_diff = torch.square(hsr_diff)
        # Sum
        return torch.sum(pos_diff)
        # # Get True/False for each pixel
        # equal_pixels = torch.isclose(raw_predictions, labels_tensor, rtol=self.rtol, atol=self.atol)
        # # equal_pixels = torch.eq(raw_predictions, labels_tensor)
        # # Count true values (sum)
        # sm = torch.sum(equal_pixels)
        # return sm


class EqualPixels2(torch.nn.Module):
    # def __init__(self, rtol=1e-05, atol=1e-08):
    def __init__(self, lambd=0.1, factor=10):
        super().__init__()
        self.factor = factor
        self.lambd = lambd
        self.remove_small = torch.nn.Hardshrink(lambd=self.lambd)
        self.to_one = torch.nn.Tanh()
        # self.rtol = rtol
        # self.atol = atol

    def forward(self, raw_predictions, labels_tensor):
        diff = torch.subtract(raw_predictions, labels_tensor)
        # Remove small values
        hsr_diff = self.remove_small(diff)
        # Get all positie values
        pos_diff = torch.abs(hsr_diff)
        # Push to 1 values that are (now) not 0
        pos_diff = torch.multiply(pos_diff, self.factor)
        pos_diff = self.to_one(pos_diff)
        # Sum
        return torch.sum(pos_diff)


class EqualPixelsBCE(torch.nn.Module):
    # def __init__(self, rtol=1e-05, atol=1e-08):
    def __init__(self, lambd=0.1, factor=10):
        super().__init__()
        self.factor = factor
        self.lambd = lambd
        self.remove_small = torch.nn.Hardshrink(lambd=self.lambd)
        self.to_one = torch.nn.Tanh()
        self.bce = torch.nn.BCELoss(reduction='none')
        # self.rtol = rtol
        # self.atol = atol

    def forward(self, raw_predictions, labels_tensor):
        diff = self.bce(raw_predictions, labels_tensor)
        # Remove small values
        hsr_diff = self.remove_small(diff)
        # # Get all positie values # Not necessary, bce always positive
        # pos_diff = torch.abs(hsr_diff)
        # Push to 1 values that are (now) not 0
        pos_diff = torch.multiply(hsr_diff, self.factor)
        pos_diff = self.to_one(pos_diff)
        # Aggregate
        return torch.mean(pos_diff)
