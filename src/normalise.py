import numpy as np
import cv2
import os
import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_darkest_frame(img_list, n = 20):
    images = []
    raw = []
    ramp = []
    raw_imgs = []
    for img in tqdm(img_list):
        if 'exposure' in img:
            pass
        else:
            continue
        rgb_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) 
        images.append(rgb_img)
        raw_img = skimage.io.imread(img.replace('jpg', 'tiff'))/65535.0
        raw_imgs.append(raw_img[::n, ::n])

    raw_imgs = np.array(raw_imgs)
    darkest_frame = raw_imgs.mean(axis = 0)

    print(darkest_frame.shape)
    np.save('darkest_frame.npy', darkest_frame)
    

def read_subtract_raw_imgs(img_list, n = 20):
    images = []
    ramp = []
    cnt = 0
    for img in tqdm(img_list):
        cnt += 1
        if 'exposure' in img:
            pass
        else:
            continue
        
        ramp_img = skimage.io.imread(img.replace('jpg', 'tiff').replace('exposure', 'ramp'))
        ramp.append(ramp_img[::n, ::n])

        if cnt == 50:
            break
    
    darkest_frame = np.load('darkest_frame.npy')
    
    raw_imgs = np.array(ramp) - darkest_frame
    
    return raw_imgs


def noise_cancel(data, c):
    n = data.shape[0]
    pixel_mean = data.mean(axis = 0)
    pixel_var = np.sum((data - pixel_mean)**2, axis = 0)/(n - 1)

    pixel_mean = pixel_mean.astype(int)
    mean_val = np.unique(pixel_mean)
    avg_val = np.array([pixel_var[pixel_mean == v].mean() for v in mean_val if v < 1500])
    mean_val = np.array([v for v in mean_val if v < 1500])
    x = mean_val
    y = avg_val
    print(len(x))
    color = {0:'r', 1:'g', 2:'b'}
    g, var_add = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.scatter(x, y, marker='.', color = color[c])
    plt.xlabel('Pixel Mean')
    plt.xlabel('Pixel Variance')
    plt.savefig(f'plot_noise_{c}.jpg')
    
    plt.clf()
    
    print(g, var_add)


def hist(data):

    i = np.random.randint(0,400)
    j = np.random.randint(0,400)

    #data[:,i,j]

    vals = data[:,i,j]
    print(vals)
    counts, bins = np.histogram(vals)
    plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.show()


n = 5
data_dir = "/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/noise"
custom_str = 'custom_out_new'
img_list = [f'{data_dir}/{x}' for x in os.listdir(data_dir) if '.jpg' in x]

raw_data = read_subtract_raw_imgs(img_list, n)
print(raw_data.shape)
noise_cancel(raw_data[:, :, :, 0], 0)
noise_cancel(raw_data[:, :, :, 1], 1)
noise_cancel(raw_data[:, :, :, 2], 2)


hist(raw_data[:, :, :, 1])