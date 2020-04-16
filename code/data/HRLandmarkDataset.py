import torch.utils.data as data
import torch
from torchvision import transforms

import numpy as np
from data import util_dataset
from PIL import Image

class HRLandmarkDataset(data.Dataset):
    '''
    Read HR images and landmark data in train and eval phases.
    '''

    def __init__(self, opt):
        super(HRLandmarkDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR = None
        
        self.norm = transforms.Normalize((0.509 * opt['rgb_range'], 
                                          0.424 * opt['rgb_range'], 
                                          0.378 * opt['rgb_range']), 
                                         (1.0, 1.0, 1.0))
        self.unnorm = transforms.Normalize((-0.509 * opt['rgb_range'], 
                                            -0.424 * opt['rgb_range'], 
                                            -0.378 * opt['rgb_range']), 
                                           (1.0, 1.0, 1.0))
        
        # read image list from image/binary files
        self.paths_HR, self.landmarks, self.bboxes = util_dataset.get_info(
            self.opt['info_path'], self.opt['dataroot_HR'])

        if 'distort' in opt.keys():
            self.distort = (opt['distort'][1] - opt['distort'][0], opt['distort'][0])
            print('Dataset distort range: %.2f, %.2f' % (opt['distort'][0], opt['distort'][1]))
        else:
            self.distort = [0, 1]
            print('Dataset no distort')
            
        assert self.paths_HR, '[Error] HR paths are empty.'

    def __getitem__(self, idx):
        hr, hr_path = self._load_file(idx) # Numpy float32, HWC, RGB, [0,255]
        landmark = self.landmarks[idx]
        bbox = self.bboxes[idx]
        distort_ratio = np.random.rand() * self.distort[0] + self.distort[1]
        hr, lr, landmark = self._crop_resize(hr, landmark, bbox, distort_ratio)
        
        # Landmark heatmap resized to 1/4 since HourGlass output small heatmaps
        landmark_resized = [(x[0] / 4, x[1] / 4) for x in landmark]
        gt_heatmap = util_dataset.generate_gt(
            (hr.shape[0] // 4, hr.shape[1] // 4), landmark_resized,
            self.opt['sigma'])  # C*H*W
        landmark = np.array(landmark)

        if self.train:
            # Landmark doesn't rotate
            lr, hr, gt_heatmap, landmark = util_dataset.augment(lr, hr, gt_heatmap, landmark, self.opt['use_flip'], self.opt['use_rot'])

        #if np.min(landmark) < 0 or np.max(landmark) >= self.opt['HR_size']:
        #    idx = torch.randint(low=0, high=len(self.paths_HR), size=(1,)).item()
        #    return self.__getitem__(idx)

        lr_tensor, hr_tensor = util_dataset.np2Tensor([lr, hr],
                                                self.opt['rgb_range'])
        lr_tensor = self.norm(lr_tensor)
        hr_tensor = self.norm(hr_tensor)
        gt_heatmap = torch.from_numpy(np.ascontiguousarray(gt_heatmap))

        return {
            'LR': lr_tensor,
            'HR': hr_tensor,
            'heatmap': gt_heatmap,
            'HR_path': hr_path,
            'landmark': landmark
        }

    def __len__(self):
        return len(self.paths_HR)

    def _load_file(self, idx):
        hr_path = self.paths_HR[idx]
        hr = util_dataset.read_img(hr_path, self.opt['data_type'])

        return hr, hr_path

    def _crop_resize(self, hr, landmark, bbox, distort_ratio=1):
        '''
        hr: HWC, RGB, [0, 255]
        landmark: list of landmarks [[x, y], ...]
        bbox: [xmin, ymin, xmax, ymax]
        distort_ratio: h/w of cropped hr image
        return:
        hr: HWC, RGB, [0, 255]
        lr: HWC, RGB, [0, 255]
        landmark: list of landmarks [[x, y], ...]
        '''
        # crop
        xmin, ymin, xmax, ymax = bbox
        # distort
        if distort_ratio != 1:
            landmark_array = np.array(landmark)
            wmin, hmin = np.max(landmark_array, axis=0) - np.min(landmark_array, axis=0)
            xmean, ymean = 0.5 * (np.max(landmark_array, axis=0) + np.min(landmark_array, axis=0))
            side_len = xmax - xmin
            # make sure crop all landmarks
            ratio = np.clip(distort_ratio, hmin / side_len, side_len / wmin)
            if ratio < 1:
                new_w = side_len
                new_h = side_len * ratio
                if ymean - 0.5 * new_h < ymin:
                    ymax = ymin + int(new_h)
                elif ymean + 0.5 * new_h > ymax:
                    ymin = ymax - int(new_h)
                else:
                    ymin = int(ymean - 0.5 * new_h)
                    ymax = int(ymean + 0.5 * new_h)
            else:
                new_w = side_len / ratio
                new_h = side_len
                if xmean - 0.5 * new_w < xmin:
                    xmax = xmin + int(new_w)
                elif xmean + 0.5 * new_w > xmax:
                    xmin = xmax - int(new_w)
                else:
                    xmin = int(xmean - 0.5 * new_w)
                    xmax = int(xmean + 0.5 * new_w)
    
        hr_cropped = hr[ymin: ymax, xmin: xmax, :]
        # resize
        img = Image.fromarray(hr_cropped.astype('uint8'))
        hr_shape = (self.opt['HR_size'], self.opt['HR_size'])
        lr_shape = (self.opt['LR_size'], self.opt['LR_size'])
        hr_img = img.resize(hr_shape, resample=Image.BICUBIC) # 128X128
        lr_img = hr_img.resize(lr_shape, resample=Image.BICUBIC) # 16X16
        # resize landmark
        w, h = xmax - xmin, ymax - ymin
        x_scale = 1.0 * self.opt['HR_size'] / w
        y_scale = 1.0 * self.opt['HR_size'] / h
        landmark_cropped = [[(p[0] - xmin) * x_scale, (p[1] - ymin) * y_scale] for p in landmark]
        return np.array(hr_img), np.array(lr_img), landmark_cropped
