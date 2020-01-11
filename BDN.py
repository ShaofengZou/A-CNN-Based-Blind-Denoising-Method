
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import time
import cv2
import numpy as np
import torch
from utils.denoising_utils import *
from common_utils import *

dtype = torch.cuda.FloatTensor
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
# Set Network
def set_network(input_depth = 3, net_type = 'unet', pad = 'reflection'):    
    # Choose network
    if net_type == 'skip':
        net = get_net(input_depth, 'skip', pad,
                      skip_n33d=128, 
                      skip_n33u=128, 
                      skip_n11=4, 
                      num_scales=5,
                      upsample_mode='bilinear').type(dtype)
    else:
        net = skip( input_depth, 3, 
                    num_channels_down = [8, 16, 32, 64, 128], 
                    num_channels_up   = [8, 16, 32, 64, 128],
                    num_channels_skip = [0, 0, 0, 4, 4], 
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    
    net = net.type(dtype)

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()])
    print ('Number of params: %d' % s)

    return net

# Denoise one image
def denoising_image(args, file, folder, img_np, img_noisy_np):

    # Set the option
    input_base_option = args.input_base_option
    input_extra_option = args.input_extra_option
    input_max_gn_std = args.input_max_gn_std
    refer_base_option = args.refer_base_option
    refer_extra_option = args.refer_extra_option
    refer_max_gn_std = args.refer_max_gn_std
    input_depth = args.input_depth
    num_iter = args.num_iter
    net_type = args.net_type

    # Set the base of input
    if input_base_option == 'use_raw_image':
        net_input_base = np_to_torch(img_noisy_np).type(dtype).detach()  
    elif input_base_option == 'use_denoised_image':       
        net_input_base = get_smooth_torch(img_noisy_np).type(dtype).detach() 
    elif input_base_option == 'use_unifrom_noise':
        net_input_base = get_noise(input_depth, 'noise' , img_np.shape[1:]).type(dtype).detach()         
    net_input_extra = net_input_base.detach().clone()
    input_std = input_max_gn_std

    # Set the base of reference
    if refer_base_option == 'use_raw_image':
        net_reference_base = np_to_torch(img_noisy_np).type(dtype).detach()  
    elif refer_base_option == 'use_denoised_image':       
        net_reference_base = get_smooth_torch(img_noisy_np).type(dtype).detach() 
    net_reference_extra = net_reference_base.detach().clone()  
    reference_std = refer_max_gn_std
    
    # Parameters
    show_iter = args.show_iter
    save_iter = args.save_iter
    LR = args.lr
    exp_weight = 0.99
     
    # Model     
    net = set_network(input_depth, net_type)  
    # net = torch.load('cc_image12.pkl')
    parameters = get_params('net', net, net_input_base)    
    optimizer = torch.optim.Adam(parameters, lr=LR)
    mse = torch.nn.MSELoss().type(dtype)

    # Variable
    psnr_gt_list = []
    psnr_gt_sm_list = []
    psnr_noisy_list = []
    ssim_gt_list = []
    ssim_gt_sm_list = []
    net_output_avg = None
    last_net = None
    psnr_noisy_last = 0     
    t1 = time.time()
    
    for i in range(1,num_iter+1):
        # Clear gradient
        optimizer.zero_grad()         

        # Input of network
        if input_extra_option == 'use_random_gaussian_std':
            input_std = np.random.uniform(0, input_max_gn_std)  
        net_input = net_input_base + (net_input_extra.normal_() * input_std / 255.0)

        # Output of network
        net_output = net(net_input)

        # Smmoth the output
        if net_output_avg is None:
            net_output_avg = net_output.detach()
        else:
            net_output_avg = net_output_avg * exp_weight + net_output.detach() * (1 - exp_weight)
        
        ## Reference of network
        if refer_extra_option == 'use_random_gaussian_std':
            reference_std = np.random.uniform(0, refer_max_gn_std)
        net_reference = net_reference_base + (net_reference_extra.normal_() * reference_std / 255.0)
        
        # Loss
        total_loss = mse(net_output, net_reference)
        total_loss.backward()
        
        # Tensor to numpy array
        output_np = torch_to_np(net_output)
        output_avg_np = torch_to_np(net_output_avg)

        # Compute psnr
        psnr_noisy = compare_psnr(img_noisy_np, output_np) 
        psnr_gt    = compare_psnr(img_np, output_np) 
        psnr_gt_sm = compare_psnr(img_np, output_avg_np) 
        psnr_noisy_list.append(psnr_noisy)
        psnr_gt_list.append(psnr_gt)
        psnr_gt_sm_list.append(psnr_gt_sm)  

        # Compute ssim
        ssim_gt = compare_ssim(X = np_transpose(img_np), Y = np_transpose(output_np), multichannel = True) 
        ssim_gt_sm = compare_ssim(X = np_transpose(img_np), Y = np_transpose(output_avg_np), multichannel = True) 
        ssim_gt_list.append(ssim_gt)
        ssim_gt_sm_list.append(ssim_gt_sm)

        # Print information
        print ('Iteration: %05d,  Loss: %f,  PSNR_noisy: %f,  PSNR_gt: %f,  PSNR_gt_sm: %f,  SSIM_gt: %f,  SSIM_gt_sm: %f' % (i, total_loss.item(), psnr_noisy, psnr_gt, psnr_gt_sm, ssim_gt, ssim_gt_sm), '\r', end='')

        # Fall back if broken
        if psnr_noisy - psnr_noisy_last < -8:
            print('Falling back to previous checkpoint')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_noisy_last = psnr_noisy

        # Save images
        filename = os.path.join(os.path.join(folder, file), str(i))
        if i % save_iter == 0:
            save_image(filename=filename, img=np_to_image(output_avg_np,True))
        
        # Show images
        if i % show_iter == 0:
            input_img = np_to_image(torch_to_np(net_input))
            refer_img = np_to_image(torch_to_np(net_reference)) 
            output_avg_img = np_to_image(output_avg_np)   
            clean_img = np_to_image(img_np)
            try:
                save_image(filename=filename, img=np.hstack((input_img, refer_img, clean_img, output_avg_img)), to_RGB=True)            
            except:
                save_image(filename=filename, img=np.hstack((refer_img, clean_img, output_avg_img)), to_RGB=True)       
            
        # Optimzer
        optimizer.step()
        
        
    # Plot psnr and ssim
    plot_psnr(folder, file, 'psnr', psnr_gt_list, psnr_gt_sm_list)
    plot_psnr(folder, file, 'ssim', ssim_gt_list, ssim_gt_sm_list)
    max_psnr = max(psnr_gt_sm_list)
    max_index = psnr_gt_sm_list.index(max_psnr)  
    
    # Print Information
    print('\n\n')
    print('============Info==============')
    print('Filename: ', file)  
    print('Result:')
    print('\t Max PSNR of smooth output: ',max_psnr)
    print('\t Index of max PSNR: ',max_index)
    print('Time cost: %.3fs'%(time.time()-t1))
    print('============End===============')
    print('\n\n')
    
    # Save psnr and ssim
    value_psnr = [psnr_gt_sm_list[i-1] for i in range(1,num_iter+1) if i % save_iter == 0]
    value_ssim = [ssim_gt_sm_list[i-1] for i in range(1,num_iter+1) if i % save_iter == 0]
    path = [os.path.join(os.path.join(folder, file), str(i) + '.jpg') for i in range(1,num_iter+1) if i % save_iter == 0]
    data = np.column_stack((np.array(path), np.array(value_psnr), np.array(value_ssim)))
    np.savetxt(os.path.join(folder,file + '.txt'), data, delimiter=" ", fmt='%s')

    return max_psnr, max_index