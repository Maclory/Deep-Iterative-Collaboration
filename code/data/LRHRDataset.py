import torch.utils.data as data
from torchvision import transforms

from data import util_dataset


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None
        
        self.norm = transforms.Normalize((0.509 * opt['rgb_range'], 
                                          0.424 * opt['rgb_range'], 
                                          0.378 * opt['rgb_range']), 
                                         (1.0, 1.0, 1.0))
        self.unnorm = transforms.Normalize((-0.509 * opt['rgb_range'], 
                                            -0.424 * opt['rgb_range'], 
                                            -0.378 * opt['rgb_range']), 
                                           (1.0, 1.0, 1.0))

        # read image list from image/binary files
        self.paths_HR = util_dataset.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR = util_dataset.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        lr_tensor, hr_tensor = util_dataset.np2Tensor([lr, hr], self.opt['rgb_range'])
        lr_tensor = self.norm(lr_tensor)
        hr_tensor = self.norm(hr_tensor)
        return {'LR': lr_tensor, 
                'HR': hr_tensor, 
                'LR_path': lr_path, 
                'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = util_dataset.read_img(lr_path, self.opt['data_type'])
        hr = util_dataset.read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path
