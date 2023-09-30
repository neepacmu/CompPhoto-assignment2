import os
import cv2
import skimage
import math
#from src.cp_exr import *
from src.cp_hw2 import *

import matplotlib.pyplot as plt
import numpy as np



def weight_tent(w_min, w_max):
    I_min, I_max = w_min*255, w_max*255
    I_max_gt = 256
    weight_vector = np.array([pixel - I_min if pixel <= 0.5*(I_max_gt - 1) else I_max_gt-pixel-1 for pixel in range(I_max_gt)])
    weight_vector[:int(I_min)] = 0
    weight_vector[int(I_max):] = 0

    return weight_vector

def weight_gaussian(w_min, w_max):
    I_min, I_max = w_min*255, w_max*255
    I_max_gt = 256

    def exp_w(I):
        return np.exp(-4*((I - 128)**2)/(16384))

    weight_vector = np.array([exp_w(pixel) for pixel in range(I_max_gt)])
    weight_vector[:int(I_min)] = 0
    weight_vector[int(I_max):] = 0

    return weight_vector

def weight_uniform(w_min, w_max):
    I_min, I_max = w_min*255, w_max*255
    I_max_gt = 256
    weight_vector = np.array([pixel for pixel in range(I_max_gt)])
    weight_vector[:int(I_min)] = 0
    weight_vector[int(I_max):] = 0

    return weight_vector

def weight(weight_type):
    z_min, z_max = 0.05, 0.95
    
    if weight_type == 'tent':
        return weight_tent(z_min, z_max)
    elif weight_type == 'gaus':
        return weight_gaussian(z_min, z_max)
    elif weight_type == 'uniform':
        return weight_uniform(z_min, z_max)


def weight_raw(I, t_k, g = 0, sigma_a = 0, weight_type = 'gaus'):
    z_min, z_max = 0.05, 0.95

    if z_min <= I and I <= z_max:

        if weight_type == 'gaus':
            return np.exp(-4*((I - 0.5)**2)/(0.25))
        elif weight_type == 'tent':
            return min(I, 1 - I)
        elif weight_type == 'uniform':
            return 1
        elif weight_type == 'photon':
            return t_k
        elif weight_type == 'optimal':
            #print("KK" , t_k)
            return (t_k**2)/(g + sigma_a)
        else:
            print("NO W")
    else:
        return 0    


def solve_non_linearity(img_list, exposures, lmd, wt_type):

    l = lmd
    pixel_range = 256

    B = [math.log(e) for e in exposures]
    w = weight(wt_type)
    n = 200

    img_list = [I[::n, ::n] for I in img_list]

    Z = np.array([img.flatten() for img in img_list])

    num_images, num_pixels = Z.shape 
    A = np.zeros((num_pixels*num_images+1 + pixel_range, pixel_range + num_pixels), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)

    row_val = 0
    for pixel_i in range(num_pixels):
        for image_j in range(num_images):
            pixel_val = Z[image_j][pixel_i]
            if wt_type == 'photon':
                weight_ij = B[image_j]
            elif wt_type == 'optimal':
                weight_ij = B[image_j]
            else:
                weight_ij = w[pixel_val]
            A[row_val][pixel_val] = weight_ij
            A[row_val][pixel_range+pixel_i] = -weight_ij
            b[row_val] = weight_ij*B[image_j]
            row_val += 1
    
    A[row_val][127] = 1
    row_val += 1

    #Regularizer
    for i in range(pixel_range-1):
        pixel_val = i
        if wt_type == 'photon':
                weight_ij = B[image_j]
        else:
            weight_ij = w[pixel_val]
        A[row_val][i]   =    l*weight_ij
        A[row_val][i+1] = -2*l*weight_ij
        A[row_val][i+2] =    l*weight_ij
        row_val += 1

    
    ans = np.linalg.lstsq(A, b)
    g = ans[0][:256]
    lE = ans[0][256:]

    print(g.shape)
    return g, lE


def get_non_linear_images(img_list, g, exposures, f_type, w_type, g_noise = None, sigma_a = None, inp_exp = 1, dark_image = None, is_raw = False):
    z_min, z_max = 12, 243
    constant_eps = 0
    log_exposure_times = [math.log(e) for e in exposures]
    
    w = weight(w_type)

    num_images = len(img_list)

    new_imgs = np.zeros_like(img_list[0], dtype='float32')

    linear_imgs = []
    weight_sums = np.zeros_like(img_list[0], dtype='float32')


    for k in range(len(img_list)):
        print(k)
        new_imgs = np.zeros_like(img_list[k], dtype='float32')
        for i in range(img_list[k].shape[0]):
            for j in range(img_list[k].shape[1]):
                for c in range(3):
                    if is_raw:
                        Ijk = img_list[k][i,j,c]
                        linear_pixel = Ijk
                    else: 
                        Ijk = int(img_list[k][i,j,c])
                    
                        linear_pixel = np.exp(g[int(Ijk)])
                        Ijk = Ijk/255.0
                        
                    
                    if w_type == 'photon':
                        weight_val = exposures[k]                        
                    elif w_type == 'optimal':
                        weight_val = (exposures[k]**2)/(g_noise[c] + sigma_a[c])
                    else:
                        weight_val = weight_raw(Ijk, exposures[k], w_type)


                    if f_type == 'log':
                        linear_pixel = Ijk + constant_eps - log_exposure_times[k]
                    else:
                        linear_pixel = Ijk/exposures[k]


                    
                    new_imgs[i,j,c] = linear_pixel * weight_val
                    weight_sums[i,j,c] += weight_val
                
        linear_imgs.append(new_imgs)
    linear_imgs = np.array(linear_imgs)

    output = linear_imgs.sum(axis = 0)/(weight_sums)
    
    #output[np.isnan(output)] = 0.05
    #output = np.clip(output,0,1)
    print(output.max(), output.min())
    return output


def read_rendered_image_exposure(img_list):
    images = []
    exposure = []
    raw = []
    k_min = 0
    k_max = 18
    for i in range(16):
        for img in img_list:
            
            k = int(img.split('.')[0].split('exposure')[-1])
            if k == i+1:
                
                if k < k_min or k > k_max:
                    continue
                rgb_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) 
                images.append(rgb_img)
                exposure.append((1/2048.0)*2**(k-1))
                raw_img = skimage.io.imread(img.replace('jpg', 'tiff'))
                
                raw.append(raw_img/65535.0)

        
                print(k)

    print(exposure)
    return images, exposure, raw


def tonemap(I_hdr):
    K = 0.10
    B = 0.95

    out = np.zeros_like(I_hdr)

    out[:,:,0] = tonemapping(I_hdr[:,:,0], K, B)
    out[:,:,1] = tonemapping(I_hdr[:,:,1], K, B)
    out[:,:,2] = tonemapping(I_hdr[:,:,2], K, B)

    out = tonemapping(I_hdr, K, B)
    #print(out)
    return out
    writeHDR('test_tone_mapped_rgb_05.HDR', out)

def hdr():
    dirname = '/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/data/door_stack'
    custom_str = 'assg'
    img_list = [f'{dirname}/{x}' for x in os.listdir(dirname) if '.jpg' in x]


    # dirname = '/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/custom_tether'
    # custom_str = 'custom_out_tether'
    # img_list = [f'{dirname}/{x}' for x in os.listdir(dirname) if '.jpg' in x]

    # dirname = '/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/new_test'
    # custom_str = 'custom_out_new'
    # img_list = [f'{dirname}/{x}' for x in os.listdir(dirname) if '.jpg' in x]

    n = 2
    lmd = 200
    raw = True

    g_noise = [8.5306, 7.2920, 0.0723]
    sigma_a = [154.24, 124.86, 53.42]

    g_noise = [10.8245, 2.2269, 4.5565]
    sigma_a = [-871.71, -60.2584, -17.174]
    dark_image = np.load('darkest_frame.npy')
    exp_noise = 1/8.0

    #weight_types = ['uniform', 'gaus' , 'tent', 'photon']
    #f_types = ['linear', 'log']
    
    weight_types = ['tent']
    f_types = ['linear', 'log']
    
    img_list, exposure_times, raw_img_list = read_rendered_image_exposure(img_list)

        
    raw_img_list = [I[::n, ::n] for I in raw_img_list]
    copy_img_list = [I[::n, ::n] for I in img_list]

    dark_image = dark_image[::1, ::1]
    
    #raw_img_list =np.array([raw_img_list[k]*65535 - (exposure_times[k]/exp_noise)*dark_image for k in range(len(raw_img_list))])/65535.0
    
    
    #print(raw_img_list.shape)
    for f_type in f_types:
        for weight_type in weight_types:
            if not raw:
                

                g, _ = solve_non_linearity(img_list, exposure_times, lmd, weight_type)
                
                plt.plot(g[12:242], 'bx')
                plt.xlabel('pixel value Z')
                plt.ylabel('log exposure X')
                plt.title(f'{weight_type} weight response curve')

                linear_rendered_images = get_non_linear_images(
                    copy_img_list, g,  exposure_times, f_type, weight_type, 
                    g_noise=g_noise,
                    sigma_a=sigma_a,
                    inp_exp=exp_noise,
                    dark_image=dark_image
                    )

                print(f'{custom_str}/out_jpg_{f_type}_{weight_type}.HDR')
                writeHDR(f'{custom_str}/out_jpg_{f_type}_{weight_type}.HDR', linear_rendered_images)


            else:
                linear_rendered_images = get_non_linear_images(
                    raw_img_list, None, exposure_times, f_type, weight_type, is_raw=True,
                    g_noise=g_noise,
                    sigma_a=sigma_a,
                    inp_exp=exp_noise,
                    dark_image=dark_image
                    )
                
                linear_rendered_images = cv2.normalize(linear_rendered_images,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    

                print(f'{custom_str}/out_raw_{f_type}_{weight_type}.HDR')
                writeHDR(f'{custom_str}/out_raw_{f_type}_{weight_type}.HDR', linear_rendered_images)

        

def geo_mean_overflow(iterable, eps = 1e-7):
    return np.exp(np.log(iterable + eps).mean())

def tonemapping(I_hdr, K, B):
    
    im_hdr = geo_mean_overflow(I_hdr)
    #print(im_hdr)
    I_hdr_hat = (K*I_hdr)/im_hdr
    #print(I_hdr_hat)
    I_white = B * np.max(I_hdr_hat)

    term1 = I_hdr_hat*(1 + (I_hdr_hat)/(I_white*I_white)) / (1 + I_hdr_hat)

    return term1


def XYZ2xyY(XYZ):

    X = XYZ[:,:,0]
    Y = XYZ[:,:,1]
    Z = XYZ[:,:,2]

    x = X/(X + Y + Z)
    y = Y/(Y+X + Z)
    Y = Y

    return np.stack([x, y, Y],axis=2)

def tonemapping_xyz(I_hdr, K, B):
    
    I_XYZ = lRGB2XYZ(I_hdr)
    I_xyY = XYZ2xyY(I_XYZ)

    Y = I_xyY[:,:,2]

    im_hdr = geo_mean_overflow(Y)
    print(im_hdr.shape)
    I_hdr_hat = (K*Y)/im_hdr

    print(I_hdr_hat.shape)
    I_white = B * np.max(I_hdr_hat)

    Y = I_hdr_hat*(1 + (I_hdr_hat)/(I_white*I_white)) / (1 + I_hdr_hat)

    x, y = I_xyY[:,:,0], I_xyY[:,:,1]

    print(x.shape, y.shape, Y.shape)
    xyY = np.stack([x, y, Y], axis = 2)

    X, Y, Z = xyY_to_XYZ(x, y, Y)

    XYZ = np.stack([X, Y, Y], axis = 2)

    out_rgb = XYZ2lRGB(XYZ)
    return out_rgb


#hdr()

I_hdr = readHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/custom_out_new/out_raw_linear_photon.HDR')

out = tonemapping(I_hdr, K = 0.05, B = 0.95)

writeHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/test_05_rgb.HDR', out)


out = tonemapping_xyz(I_hdr, K = 0.05, B = 0.95)

writeHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/test_05_xyy.HDR', out)

I_hdr = readHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/custom_out_new/out_raw_linear_photon.HDR')

out = tonemapping(I_hdr, K = 0.15, B = 0.95)

writeHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/test_15_rgb.HDR', out)


out = tonemapping_xyz(I_hdr, K = 0.15, B = 0.95)

writeHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/test_15_xyy.HDR', out)


# # lRGB2XYZ(I_hdr)