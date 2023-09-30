import matplotlib.pyplot as plt
from src.cp_hw2 import *
import cv2

# img = readHDR('out_log.HDR')
# plt.imshow(img)
# temp = plt.ginput(48, timeout=-1)
# np.save('temp.npy', temp)
# print(temp)

def white_balance_patch(data, patch_1, patch_2):
    x1, y1 = int(patch_1[0]), int(patch_1[1])
    x2, y2 = int(patch_2[0]), int(patch_2[1])

    print(x1, x2, y1, y2)
    patch = data[y1:y2, x1:x2]
    R, G, B = patch[:,:,0], patch[:,:,1], patch[:,:,2]

    print("KK", patch.shape)
    r_val = R.mean()
    g_val = G.mean()
    b_val = B.mean()

    
    print(r_val, g_val, b_val)
    R = data[:,:,0]/r_val
    G = data[:,:,1]/g_val
    B = data[:,:,2]/b_val

    return np.stack([R, G, B],axis=2)


def estimate_homography(points1, points2):

    A = []

    for p1,p2 in zip(points1, points2):
        print(p1, p2)
        x1, y1, _ = p1
        x2, y2, _ = p2

        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    A = np.array(A)

    U, S, V = np.linalg.svd(A)

    H = V[-1].reshape((3,3))

    return H


hdr_file_path = '/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/out_raw_linear_gaus.HDR'

img = readHDR(hdr_file_path)
img_test = cv2.imread(hdr_file_path, flags=cv2.IMREAD_ANYDEPTH)
coords = np.load('coordinates.npy')

GT = read_colorchecker_gm()

mean_gt = np.stack((GT[0].flatten(), GT[1].flatten(), GT[2].flatten(), np.ones_like(GT[0].flatten()))).T

print(mean_gt.shape, GT[0].shape, len(GT))

curr_rgb_vals = []

for i in range(24):

    coord_min, coord_max = coords[2*i], coords[2*i+1]

    coords_mid = (coord_min + coord_max)/2

    cv2.putText(img_test, str(i),  (int(coords_mid[0]), int(coords_mid[1])),cv2.FONT_HERSHEY_SIMPLEX, 1,   (255,50, 0), 1, 1)

    img_patch = img[int(coord_min[1]):int(coord_max[1]), int(coord_min[0]):int(coord_max[0])]

    temp = img_patch.mean(axis = (0,1))

    temp = img[int(coords_mid[1]), int(coords_mid[0])]



    curr_rgb_vals.append([temp[0], temp[1], temp[2], 1])


writeHDR('test.HDR', img_test[:,:,:3])

curr_rgb_vals = np.array(curr_rgb_vals)


A = curr_rgb_vals
Y = mean_gt

checker_coords_max, checker_coords_min = coords[18], coords[5]

source_image = img

X, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

temp_Y = np.dot(curr_rgb_vals,X)

image_flattened = source_image.reshape(-1,3)

image_homogen = np.concatenate([image_flattened, np.ones_like(image_flattened[:,0:1])], axis = -1)

processed_image = np.dot(image_homogen,X)

img_out = processed_image[:,:3].reshape(source_image.shape)
#img_out = np.clip(img_out,0,1)
img_out[img_out < 0] = 0

white_start = coords[2*18]
white_end = coords[2*18 + 1]

img_out = white_balance_patch(img_out, white_start, white_end)

print(img_out.shape)
writeHDR('/Users/neerajpanse/np/CMU/Fall-23/CompPhoto/assgn2/assg/color_corrected.HDR', img_out[:,:,:3])
