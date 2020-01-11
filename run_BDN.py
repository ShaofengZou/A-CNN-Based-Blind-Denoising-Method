import argparse
'''Usage
python run_BDN.py 
'''
parser = argparse.ArgumentParser(description="Training Blind Image Quality Assessment Network")
parser.add_argument('-id', '--script-id', type=str, default='BlindDenoising', 
                    help="ID of this experiment")

parser.add_argument("--dataset_real_path", type=str, default='dataset/CC/real/',
                    help='folder of noisy images')
parser.add_argument("--dataset_mean_path", type=str, default='dataset/CC/mean/',
                    help='folder of clean images')
parser.add_argument("--output_path", type=str, default='dataset/CC/result',
                    help="output dir")

parser.add_argument("--is_split_dataset", type=bool, default=False,
                    help='split the dataset to several one and use multiple gpu to train')
parser.add_argument("--dataset_part_id", type=str, default='1-7-',
                    help='number of time the learning of each training stage')


parser.add_argument("--lr", type=float, default=1e-2,
                    help='learning rate of training')
parser.add_argument("--num_iter", type=int, default=3000,
                    help="number of iteration for each image")    
parser.add_argument("--input_depth", type=int, default=32,
                    help="number of input channels")        
parser.add_argument("--net_type", type=str, default='skip',  
                    help="type of network")

parser.add_argument("--output_checkpoint", type=str, default='checkpoint',
                    help="checkpoint dir")

parser.add_argument("--input_base_option", type=str, default='use_unifrom_noise', 
                    choices=['use_raw_image', 'use_denoised_image', 'use_unifrom_noise'],
                    help="option for base of input")
parser.add_argument("--input_extra_option", type=str, default='use_fix_gaussian_std', 
                    choices=['use_random_gaussian_std', 'use_fix_gaussian_std', 'None'],
                    help="option for extra of input")
parser.add_argument("--input_max_gn_std", type=int, default=8.5, 
                    help="max std of input gaussian noise")

parser.add_argument("--refer_base_option", type=str, default='use_raw_image', 
                    choices=['use_raw_image', 'use_denoised_image'],
                    help="option for base of reference")
parser.add_argument("--refer_extra_option", type=str, default='None', 
                    choices=['use_random_gaussian_std', 'use_fix_gaussian_std', 'None'],
                    help="option for base of reference")
parser.add_argument("--refer_max_gn_std", type=int, default=8.5,                
                    help="max std of reference gaussian noise")

parser.add_argument("--show_iter", type=int, default=1000,                
                    help="the interval of showing the reconstructed result")
parser.add_argument("--save_iter", type=int, default=100,                
                    help="the interval of saving the reconstructed result")

parser.add_argument("--gpu_id", type=int, default=1,
                    help="the id of gpu card")
 

args = parser.parse_args()

print('Start up Script')
print("\n================= Training Blind Image Quality Assessment Network =================")
print("> Parameters:")
for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))


import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import glob
import os
import cv2
import time
import matplotlib.pyplot as plt
import warnings

import torch
 
from models import *
from utils.denoising_utils import *
from common_utils import *
from BDN import *

warnings.filterwarnings("ignore")
 
# Set path of image folder 
dataset_real_path = args.dataset_real_path
dataset_mean_path = args.dataset_mean_path
output_path =  args.output_path

# Make dirs
os.makedirs(output_path, exist_ok=True)

# Read pair of nosiy and clean images
real_files = glob_image_files(dataset_real_path)
mean_files = glob_image_files(dataset_mean_path)
real_files = check_and_generate_noisy_image(real_files, mean_files, dataset_real_path)
print('\n========Dataset Info========')
print('Size of real images dataset:',len(real_files))
print('Part of real images dataset:\n',real_files[:5],'\r')
print('Size of mean images dataset:',len(mean_files))
print('Part of mean images dataset:\n',mean_files[:5],'\r')

# Get only part of dataset for multi GPU training
if args.is_split_dataset:
    dataset_part = args.dataset_part_id.split('/')[0]
    total_dataset_part = args.dataset_part_id.split('/')[1]
    real_files, mean_files = get_part_dataset(real_files, mean_files, dataset_part, total_dataset_part)
    print('After split dataset: mean images:',mean_files)
    print('After split dataset: real images:',real_files)
else:
    args.dataset_part_id = ''
print('='*30, '\n')

# Parameters
num_iter = args.num_iter
imsize =-1
input_depth = args.input_depth
net_type = args.net_type

# Variable
max_psnr_list = []
max_index_list = []

for index, (real_file, mean_file) in enumerate(zip(real_files, mean_files)): 
    # Print information
    # print('\n')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Processing: ', index+1, '/', len(real_files))
    print('Current pair of image is: %s + %s'%(real_file, mean_file))

    # Get real noisy image as reference and mean image as ground truth,  np C x W x H
    mean_img_pil = crop_image(get_image(mean_file, imsize)[0], d=32)
    mean_img_np = pil_to_np(mean_img_pil)
    real_img_pil = crop_image(get_image(real_file, imsize)[0], d=32)
    real_img_np = pil_to_np(real_img_pil)
    
    # Makedir
    file_suffix =  real_file.split('/')[-1].split('.')[0] # such as dataset/CC/real/5dmark3_iso3200_1_real.png -> 5dmark3_iso3200_1_real
    file_target_name = file_suffix    
    os.makedirs(os.path.join(output_path, file_target_name), exist_ok=True) 

    # Start blind denoising
    max_psnr, max_index = denoising_image(args, file_target_name, output_path, mean_img_np, real_img_np)

    # Save max psnr and index
    max_psnr_list.append(max_psnr)
    max_index_list.append(max_index)
    np.savetxt(os.path.join(output_path, args.dataset_part_id  + 'best_psnr.txt'), max_psnr_list, delimiter=" ", fmt='%s')
    np.savetxt(os.path.join(output_path, args.dataset_part_id  + 'best_psnr_index.txt'), max_index_list, delimiter=" ", fmt='%s')
    