import os
import math
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

####################
# miscellaneous
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        choice = input('Are you sure? y/[n]')
        if choice is not 'y':
            print('Give up renaming, exit')
            exit(0)
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
def Tensor2np(tensor_list, rgb_range):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()


####################
# metric
####################
def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        
####################
# landmark
####################

def calc_nme(output, target, norm='inter-ocular', norm_size=None):
    '''
    output: (B, N, 2)
    target: (B, N, 2)
    norm: 'inter-ocular' or 'bbox'
    norm_size: size of bbox if norm is 'bbox'
    ''' 
    assert target.shape == output.shape
    batch_size, num_keypoints, _ = target.shape
    if norm == 'inter-ocular':
        if num_keypoints == 5:
            norm_size = target[:, 0, :] - target[:, 1, :]
        elif num_keypoints == 68:
            norm_size = target[:, 37, :] - target[:, 46, :]
        else:
            raise NotImplementedError('Key point number not implemented!')
        norm_size = np.sqrt(np.sum(np.square(norm_size), axis=1)) # (B, 1)
        norm_size = np.reshape(norm_size, (batch_size, 1))
    elif norm =='bbox':
        assert norm_size != None
        norm_size = np.array(norm_size)
    diff = target - output
    diff = np.sqrt(np.sum(np.square(diff), axis=2)) # (B, N)
    diff /= norm_size
    diff = np.mean(diff, axis=(0, 1))
    return diff

def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N, 5, 32, 32)
    :return: numpy array (N, 5, 2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def plot_heatmap_compare(heatmap, heatmap_gt, img, img_gt, scale=4, alpha=0.5):
    '''
    merge heatmaps of different points into one heatmap
    :param heatmap: list of numpy array (5, 32, 32)
    :param heatmap_gt: numpy array (5, 32, 32) ground truth
    :param img: image array (128, 128, 3) SR image
    :param img_gt: image array (128, 128, 3) image ground truth
    :param scale: scale factor
    :param alpha: float alpha
    '''
    heatmap_list = [heatmap_gt]
    heatmap_list.extend(heatmap)
    scaled = [merge_and_scale_heatmap(x, scale) for x in heatmap_list]
    scaled_s = np.concatenate(scaled, axis=1)
    
    img_list = [img_gt]
    img_list.extend(img)
    img_s = np.concatenate(img_list, axis=1)
    
    fig_withhm = plt.figure(figsize=(2 * len(heatmap_list), 2))
    plt.imshow(img_s)
    plt.imshow(scaled_s, cmap='hot', alpha=alpha)
    plt.axis('off')
    return fig_withhm

def plot_landmark_compare(landmark, img, img_gt):
    '''
    plot landmarks in faces
    :param landmark: list of numpy array (N, 2)
    :param img: list of image array (128, 128, 3) SR image
    :param img_gt: list of image array (128, 128, 3) image ground truth
    '''
    w, h = img[0].shape[:2]
    all_landmarks = []
    for i, l in enumerate(landmark):
        biased = l + np.array([w * (i + 1), 0]).astype(float)
        all_landmarks.append(biased)
    all_landmarks = np.concatenate(all_landmarks, axis=0)
        
    img_list = [img_gt]
    img_list.extend(img)
    img_s = np.concatenate(img_list, axis=1)
    
    fig_withlm = plt.figure(figsize=(4 * len(img_list), 4))
    plt.imshow(img_s)
    plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], linewidths=0.5, c='red')
    plt.axis('off')
    return fig_withlm

def merge_and_scale_heatmap(heatmap, scale):
    merged = np.mean(heatmap, axis=0)
    h, w = merged.shape
    scaled = cv2.resize(merged, dsize=(h * scale, w * scale), interpolation=cv2.INTER_LINEAR)
    return scaled
