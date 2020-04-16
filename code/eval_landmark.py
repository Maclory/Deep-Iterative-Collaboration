import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from networks.modules.architecture import StackedHourGlass

class ImageDataset(Dataset):
    def __init__(self, path, info_path):
        self.file_list = []
        for x in os.listdir(path):
            if x.endswith('png') or x.endswith('jpg'):
                self.file_list.append(os.path.join(path, x))
        with open(info_path, 'rb') as f:
            self.info_dict = pickle.load(f)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.509, 0.424, 0.378), (1, 1, 1))
        ])
        self.to_image = transforms.Compose([
            transforms.Normalize((-0.509, -0.424, -0.378), (1, 1, 1)),
            transforms.ToPILImage()
        ])

    def __getitem__(self, index):
        path = self.file_list[index]
        img = Image.open(path)
        name = os.path.basename(path)
        landmark = self.info_dict[name.replace('jpg', 'png')]
        return self.transform(img), landmark
    
    def __len__(self):
        return len(self.file_list)
    
def get_peak_2(heatmap_one):
    '''
    heatmap_one: 32 * 32
    '''
    h, w = heatmap_one.shape
    idx = torch.argsort(heatmap_one.view(-1), descending=True)
    top1 = (idx[0].item() // h, idx[0].item() % w)
    top2 = (idx[1].item() // h, idx[1].item() % w)
    return top1, top2

def get_peak(heatmap_one):
    top1, top2 = get_peak_2(heatmap_one)
    top1 = np.array(top1)
    top2 = np.array(top2)
    trans = (top2 - top1) > 0
    trans = trans.astype(int) * 2 - 1
    peak = top1 * 4 + trans 
    return peak[[1, 0]] # x, y

def get_landmark(heatmap):
    '''
    heatmap: 68 * 32 * 32
    '''
    landmarks = []
    num = heatmap.shape[0]
    for i in range(num):
        landmarks.append(get_peak(heatmap[i]))
        
    return np.array(landmarks)

def main(args):
    data_root = args.data_root
    info_path = args.info_path

    hg_opt = {
        "hg_num_feature": 256,
        "hg_num_stack": 4,
        "hg_num_keypoints": 68,
        "hg_connect_type": "mean"
    }
    
    ds = ImageDataset(data_root, info_path)
    dl = DataLoader(ds)
    print('Number of images %d' % len(ds))
    net = StackedHourGlass(hg_opt['hg_num_feature'], hg_opt['hg_num_stack'],
                           hg_opt['hg_num_keypoints'], hg_opt['hg_connect_type'])
    net.eval()
    net.load_state_dict(torch.load('../models/HG_68_CelebA.pth'))
    net = net.cuda()
    res_dict = {
        'width': [],
        'eye_outer_width': [],
        'eye_dist': []
    }
    
    for SR, gt_landmark in tqdm(dl, total=len(dl)):
        gt_landmark = gt_landmark[0].numpy()
        SR = SR.cuda()
        with torch.no_grad():
            heatmap, _ = net(SR, None)
            landmark = get_landmark(heatmap[0])
            diff = landmark - gt_landmark
            rmse = np.sqrt((diff**2).sum(1).mean())

            width = (np.max(gt_landmark, 0) - np.min(gt_landmark, 0))[0]
            eye_outer_width = gt_landmark[46, 0] - gt_landmark[37, 0]
            eye_dist = np.mean(gt_landmark[[43, 46], 0]) - np.mean(
                gt_landmark[[37, 40], 0])
            res_dict['width'].append(rmse / width)
            res_dict['eye_outer_width'].append(rmse / eye_outer_width)
            res_dict['eye_dist'].append(rmse / eye_dist)
            
    with open(os.path.join(data_root, 'landmark_result.json'), 'w') as f:
        json.dump(res_dict, f)
    with open(os.path.join(data_root, 'landmark_average_result.txt'), 'w') as f:
        res_str = ''
        for k, v in res_dict.items():
            res_str += '%s: %.4f\n' % (k, np.mean(v))
        f.write(res_str)
    print(res_str)
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Landmark evaluation')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--info_path', type=str)
    main(parser.parse_args())
  
