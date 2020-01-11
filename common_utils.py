import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr
from skimage.util import random_noise
import glob

from models import *
from utils.denoising_utils import *

def np_to_image(img_np, to_RGB=False):
    '''Converts image in np.array format to opencv image format BGR
    
    From C x W x H [0...1] to W x H x C [0..255]
    '''

    vis = np.transpose(img_np,(1,2,0))
    vis = vis.clip(0,1)*255
    vis = vis.astype('uint8')

    if to_RGB:
        vis = vis[:,:,::-1]
    return vis

def np_transpose(img_np):
    vis = np.transpose(img_np,(1,2,0))
    return vis

# Plot PSNR
def plot_psnr(folder, file, name,  psnr_gts, psnr_gts_sm, x_val = None, marker = None, xlabel = None, ylabel = None, alabel = 'Ground truth - Ouput', blabel = 'Ground truth smooth - Ouput', figsize = (8,4)):
    if x_val is None:
        plot_x = range(1, len(psnr_gts) + 1)
    else:
        plot_x = x_val

    plt.figure(figsize = figsize, dpi = 144)   

    if marker is None:
        plt.plot(plot_x, psnr_gts, 'b', label= alabel)
        plt.plot(plot_x, psnr_gts_sm, 'g', label= blabel)
    else:
        plt.plot(plot_x, psnr_gts, 'b',  marker='o',label= alabel)
        plt.plot(plot_x, psnr_gts_sm, 'g', marker='*', label= blabel)

    plt.title('File: ' + file + 'Metric: ' + name)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(os.path.join(folder,file + '_' + name + '.png') , bbox_inches='tight')

# Smooth the image
def get_smooth_torch(img_noisy_np):
    img = np_to_image(img_noisy_np)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    dst = np_to_torch(pil_to_np(laplacian))
    return dst
 

# Read images from image folder
def glob_image_files(dataset_path):    
    files = glob.glob(dataset_path + "*.png")
    files.extend(glob.glob(dataset_path + "*.JPG"))
    files.extend(glob.glob(dataset_path + "*.bmp"))
    files.extend(glob.glob(dataset_path + "*.BMP"))
    files = sorted(files)
    return files

# Check whether has noisy image, otherwise generate noisy images by adding AGWN
def check_and_generate_noisy_image(real_files, mean_files, dataset_real_path):
    if len(real_files) == 0:
        for file in mean_files:
            img = cv2.imread(file)    
            noise = np.random.randn(*img.shape) * 25
            noise_img = img + noise
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dataset_real_path, file.split('/')[-1].split('.')[0]+'_real.BMP'), noise_img)
            
        real_files = glob.glob(dataset_real_path + "*.png")
        real_files.extend(glob.glob(dataset_real_path + "*.JPG"))
        real_files.extend(glob.glob(dataset_real_path + "*.bmp"))
        real_files.extend(glob.glob(dataset_real_path + "*.BMP"))
        real_files = sorted(real_files)
    return real_files

# Get only part of dataset for multi GPU training
def get_part_dataset(real_files, mean_files, dataset_part, total_dataset_part):
    each_number = len(mean_files)//total_dataset_part
    mean_files = mean_files[each_number*(dataset_part-1):each_number*dataset_part]
    real_files = real_files[each_number*(dataset_part-1):each_number*dataset_part]
    return real_files, mean_files


# Plot the training histroy
def plot_history(history, save_name):
    plt.figure(dpi=144)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)

    plt.figure(dpi=144)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_name + '.png', bbox_inches='tight')

# Plot list
def plot_list(x=None, y=None, title=None, label=None, filename=None, color='b'):
    y = np.array(y)
    if x == None:
        x = range(1, len(y) + 1)

    plt.figure(dpi = 144) 
    plt.plot(x, y, color, label = label)
    plt.title(title)
    plt.legend()

    if filename!=None:
        plt.savefig(filename + '.png')

# Save image
def save_image(filename=None, img=None, to_RGB=False):
    if to_RGB:
        cv2.imwrite(filename + '.png', img[:,:,::-1])
    else:
        cv2.imwrite(filename + '.png', img)
        
# Return max value and corresponding index from a list
def get_max_value(psnrs):
    max_value = max(psnrs)
    max_value_index = psnrs.index(max_value)
    return max_value, max_value_index

# Get mean of pnsrs
def mean_psnr(psnrs):    
    return np.mean(np.array(psnrs))


# Calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# Calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std