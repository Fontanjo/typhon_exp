import torch
from typhon_model import TyphonModel
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time


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

	# Add batch channel and color channel
	input.shape = (1,1, *input.shape)

	input_tensor = torch.from_numpy(input).to(cuda_device)

	output = model(input_tensor.float(), 'BraTS2019_LGG_flair')

	# Remove dimensions of size 1
	npinput = input.squeeze()
	npmask = mask.squeeze()
	npoutput = output.detach().numpy().squeeze()

	# Uniform mask (as for metrics calculation)
	npoutput_unary = np.zeros((npoutput.shape))
	npoutput_unary[npoutput > 0.5] = 1


	# data is your numpy array with shape (617, 767)
	# orig_img = Image.fromarray(npinput.dot([0.299, 0.587, 0.114]) * 255) # For some reason can't show as colored mri, show as grayscale 
	orig_img = Image.fromarray(npinput * 255)
	out_img = Image.fromarray(npoutput * 255)
	mask_img = Image.fromarray(npmask * 255)
	out_unary_img = Image.fromarray(npoutput_unary * 255)

	# Show images
	out_img.show(title='output')
	time.sleep(1)
	mask_img.show(title='mask')
	time.sleep(0.1)
	orig_img.show(title='orig')
	time.sleep(0.1)
	out_unary_img.show(title='output unary')
