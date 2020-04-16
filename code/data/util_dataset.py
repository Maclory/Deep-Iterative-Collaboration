import os, json, random, pickle
import numpy as np
import cv2
import torch

####################
# Files & IO
####################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_img(path, data_type):
    # read image by cv2 or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if data_type == 'img':
        img = cv2.imread(path)[:, :, ::-1]
    elif data_type.find('npy') >= 0:
        img = np.load(path)
    else:
        raise NotImplementedError

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths

def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]

####################
# Landmark
####################
def get_landmark(landmark_dict_path, paths_HR):
    landmark_list = []
    with open(landmark_dict_path, 'r') as f:
        landmark_dict = json.load(f)
#    import pdb; pdb.set_trace()
    for p in paths_HR:
        img_name = os.path.basename(os.path.splitext(p)[0])
        landmark_list.append(landmark_dict[img_name])
    return landmark_list

# read infomation(bbox and landmark)
def get_info(info_path, root_path):
    path_list = []
    landmark_list = []
    bbox_list = []
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)
    for img_name, info in info_dict.items():
        path_list.append(os.path.join(root_path, img_name))
        landmark_list.append(info['landmark'])
        bbox_list.append(info['bbox'])
    return path_list, landmark_list, bbox_list

def generate_gt(size, landmark_list, sigma):
    '''
    return N * H * W
    '''
    heatmap_list = [
        _generate_one_heatmap(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

# def augment(lr, hr, gt, landmark, hflip):
#     def _augment(img):
#         if hflip: img = img[:, ::-1, :]
#         if vflip: img = img[::-1, :, :]
#         if rot90: img = img.transpose(1, 0, 2)
#         return img
    
#     if random.random() > 0.5 and hflip:
#         lr = lr[:, ::-1, :]
#         hr = hr[:, ::-1, :]
#         gt = _hflip_heatmap_channels(gt)[:, :, ::-1]
#         landmark[:, 0] = hr.shape[1] - landmark[:, 0]
#     return lr, hr, gt, landmark

def augment(lr, hr, gt, landmark, hflip, rot):
    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
    
    if random.random() > 0.5 and hflip:
        lr = lr[:, ::-1, :]
        hr = hr[:, ::-1, :]
        gt = _hflip_heatmap_channels(gt)[:, :, ::-1]
        landmark[:, 0] = hr.shape[1] - landmark[:, 0]
    if rot:
        rot_rand = random.random()
        if rot_rand > 0.75:
            lr = np.rot90(lr, k=1, axes=(0, 1))
            hr = np.rot90(hr, k=1, axes=(0, 1))
            gt = np.rot90(gt, k=1, axes=(1, 2))
        elif rot_rand > 0.5:
            lr = np.rot90(lr, k=2, axes=(0, 1))
            hr = np.rot90(hr, k=2, axes=(0, 1))
            gt = np.rot90(gt, k=2, axes=(1, 2))
        elif rot_rand > 0.25:
            lr = np.rot90(lr, k=3, axes=(0, 1))
            hr = np.rot90(hr, k=3, axes=(0, 1))
            gt = np.rot90(gt, k=3, axes=(1, 2))
    return lr, hr, gt, landmark

def _hflip_heatmap_channels(heatmap):
    '''
    heatmap: B*W*H
    '''
    num_keypoints = heatmap.shape[0]
    if num_keypoints == 68:
        def reverse_round(all_indices):
            assert all_indices.shape[0] % 2 == 0
            mid_index = int((all_indices[-1] - all_indices[0] + 1) / 2)
            return_indices = np.zeros_like(all_indices)
            # upper part
            return_indices[:mid_index+1] = all_indices[:mid_index+1][::-1]
            return_indices[mid_index+1:] = all_indices[mid_index+1:][::-1]
            return return_indices
        
        index = np.arange(start=0, stop=68, dtype=int)
        # face outline
        index[:17] = index[:17][::-1]
        # mouth
        index[48: 60] = reverse_round(index[48: 60])
        index[60: 68] = reverse_round(index[60: 68])
        # eyes
        index[36: 42] = reverse_round(index[36: 42])
        index[42: 48] = reverse_round(index[42: 48])
        # eyebrow
        index[17: 27] = index[17: 27][::-1]
        # nose
        index[31: 36] = index[31: 36][::-1]
    else:
        raise NotImplementedError('Horizontal flip for heatmap is not implemented for %d points' % num_keypoints)
        
    return heatmap[index, :, :]
