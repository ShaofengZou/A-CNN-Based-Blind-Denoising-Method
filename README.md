# A CNN-Based Blind Denoising Method

For blind denoising, we optimize the [Deep Image Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) method by transfer learning to reduce the number of iteration. 

To determine the quality of reconstructed image, the blind image assessment network based on MobileNet is presented to estimate the scores of image quality. 

The experimental results show that our proposed method has a good noise suppression for the enhanced dark endoscopic images from the view of the visual quality.



## Architecture

![architecture](README\architecture.png)

## Dependencies

* torch (1.0.1.post2)
* tensorflow (1.12.0)
* keras (2.1.6)
* numpy
* matplotlib
* skimage
* opencv-python
* pandas

Run on ubuntu 16.04, python3.5, CUDA 9.0, cuDNN 7.5 with 8 Nvidia TITAN Xp  GPUs.



## Usage

### Train blind denoising network(BDN) to reconstruct noisy image

##### Step1. Reconstruct PolyU dataset

> python run_BDN.py   --dataset_real_path dataset/CC/real/ \
>
> ​         							 --dataset_mean_path dataset/CC/mean/ \
>
> ​         							 --output_path dataset/CC/20200111 \
>
> ​          							--num_iter 3500 --save_iter 20 --lr 1e-2 --gpu_id 1

##### Step2. Reconstruct CC dataset

> python run_BDN.py   --dataset_real_path dataset/PolyU/real/ \
>
> ​          							--dataset_mean_path dataset/PolyU/mean/ \
>
> ​          							--output_path dataset/PolyU/20200111 \
>
> ​          							--num_iter 5000 --save_iter 20 --lr 1e-2 --gpu_id 2

##### Tips

[PolyU dataset](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) has 100 pairs of noisy and clean images and [CC dataset](https://github.com/woozzu/ccnoise) has 15 pairs of noisy and clean images.

Use the method of deep image prior to reconstruct one image will cost about 15 minutes with 3000 iteration. 

So you can simply download the reconstructed images from [here](https://cloud.tsinghua.edu.cn/d/a831977f645f4cfa94a3/) or [here](https://pan.baidu.com/s/1KHsomll7PkwF8aM8IwQTpA) and then update the dataset.



### Train blind image quality assessment network(BIQAN)

##### Setp1. Convert the PSNR of reconstructed images to AVA dataset format

> python generator_dataset.py  --dataset_path dataset/PolyU_Mulit_UN_GN \
>
> ​               										--output_path dataset/PolyU_Mulit_UN_GN_PNSR.txt

##### Step2. Train the blind image quality assessment network

> python train_BIQAN.py   --dataset_image_path dataset/PolyU/PolyU_Mulit_UN_GN\
>
> ​            								--dataset_file_path dataset/PolyU/PolyU_Mulit_UN_GN_PNSR.txt\
>
> ​            								--output_checkpoint checkpoint/PolyU_Mulit_UN_GN/mobilenet\
>
> ​            								--output_path result/PolyU_Mulit_UN_GN\
>
> ​            								--epochs 40 --lr 1e-3 --gpu_id 3

##### Tips

You can download the trained model from [here](https://cloud.tsinghua.edu.cn/d/18d86b3a04c04b80a41f/) or [here](https://pan.baidu.com/s/1S8gzKCiDBE8VbNhTRSZngA)

##### Step3. Test the blind image quality assessment network

> python test_BIQAN.py --dataset_image_path dataset/CC/CC_Resume_All32 \
>
> ​            							--pre_trained_model checkpoint/PolyU_Mulit_UN_GN/mobilenet/weights.004-0.046.hdf5 \
>
> ​           							 --output_path result/CC_Resume_All32 \
>
> ​           							 --start_index 90 --gpu_id 4



## Results

#### Some reconstructed images

![1](README\result1.png)

#### Mobilenet training loss

<img src="result\PolyU_Mulit_UN_GN\first stage.png" alt="first stage" style="zoom: 80%;" />

#### Best reconstruct image and denoised image choosen by BIQAN

<img src="result\CC_Resume_All32\d800_iso1600_2_Compare.png" alt="d800_iso1600_2_Compare" style="zoom:80%;" />



## Reference

Part of code refers from [deep image prior](https://github.com/DmitryUlyanov/deep-image-prior) and [neural-image-assessment](https://github.com/titu1994/neural-image-assessment).