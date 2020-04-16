import torch.utils.data as data
from torchvision import transforms

from data import util_dataset


class LRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        
        self.norm = transforms.Normalize((0.509 * opt['rgb_range'], 
                                          0.424 * opt['rgb_range'], 
                                          0.378 * opt['rgb_range']), 
                                         (1.0, 1.0, 1.0))
        self.unnorm = transforms.Normalize((-0.509 * opt['rgb_range'], 
                                            -0.424 * opt['rgb_range'], 
                                            -0.378 * opt['rgb_range']), 
                                           (1.0, 1.0, 1.0))

        # read image list from image/binary files
        self.paths_LR = util_dataset.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

    def __getitem__(self, idx):
        lr, lr_path = self._load_file(idx)
        lr_tensor = util_dataset.np2Tensor([lr], self.opt['rgb_range'])[0]
        lr_tensor = self.norm(lr_tensor)
        return {'LR': lr_tensor, 
                'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)


    def _get_index(self, idx):
        return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        lr = util_dataset.read_img(lr_path, self.opt['data_type'])

        return lr, lr_path
