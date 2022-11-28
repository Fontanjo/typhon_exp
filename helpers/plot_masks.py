import torch
from typhon_model import TyphonModel
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

# img_name = "test/TCGA_CS_4944_20010208_17" # No
# img_name = "test/TCGA_DU_5849_19950405_29" # Yes
# img_name = "test/TCGA_DU_5852_19950709_35" # No
# img_name = "test/TCGA_DU_5853_19950823_22" # No
# img_name = "test/TCGA_DU_5871_19941206_34" # No
#
# img_name = "train/TCGA_DU_8165_19970205_8" # No
# img_name = "train/TCGA_DU_8165_19970205_29" # No
# img_name = "train/TCGA_DU_8166_19970322_5" # No
# img_name = "train/TCGA_DU_8167_19970402_4" # No
# img_name = "train/TCGA_DU_8168_19970503_31" # No
#
# img_name = "val/TCGA_CS_4941_19960909_11" # Yes
# img_name = "val/TCGA_CS_4941_19960909_13" # Yes
# img_name = "val/TCGA_CS_4942_19970222_10" # Yes
# img_name = "val/TCGA_CS_4943_20000902_11" # No
# img_name = "val/TCGA_CS_4944_20010208_1" # No


# model_path = "./results/20221005_test_segmentation_spec_50_freq_1_lr_bigger_gpu_0/models/spec_model_Brain_p.pth"
# model_path = "./results/20221005_test_segmentation_spec_300_freq_1_lr_bigger_gpu_0/models/spec_model_Brain_p.pth"
# input_path = f"./datasets_segmentation/Brain/{img_name}.npy"



model_path = '/home/jonas/Documents/master_thesis/codes/typhon_exp/results/20221115_BraTS_LGG_iou_long_0/models/train_model_p.pth'
input_path = '/home/jonas/Documents/master_thesis/codes/typhon_exp/datasets_mini/BraTS2019_LGG_flair/test/BraTS19_TMC_09043_1_slice_78_flair.npy'




cuda_device = "cpu"


if __name__ == "__main__":
	loaded_state_dicts = torch.load(model_path, map_location=cuda_device)
	dsets_names = loaded_state_dicts['variables']['dsets_names']
	model = TyphonModel.from_state_dict(loaded_state_dicts)

	mask_path = input_path[:-4] + "_mask.npy"

	model.to(cuda_device)

	input = np.load(input_path)
	mask = np.load(mask_path)

	# Add batch channel
	input.shape = (1, *input.shape)

	input_tensor = torch.from_numpy(input).to(cuda_device)

	output = model(input_tensor.float(), 'Brain')

	# Remove dimensions of size 1
	npinput = input.squeeze()
	npinput = npinput.transpose(1, 2, 0)
	npmask = mask.squeeze()
	npoutput = output.detach().numpy().squeeze()


	# data is your numpy array with shape (617, 767)
	orig_img = Image.fromarray(npinput.dot([0.299, 0.587, 0.114]) * 255) # For some reason can't show as colored mri, show as grayscale # TODO fix
	out_img = Image.fromarray(npoutput * 255)
	mask_img = Image.fromarray(npmask * 255)

	# Show images
	out_img.show(title='output')
	time.sleep(1)
	mask_img.show(title='mask')
	time.sleep(0.1)
	orig_img.show(title='orig')
