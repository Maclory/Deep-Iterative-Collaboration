import torch.utils.data

def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LRHR':
        from data.LRHRDataset import LRHRDataset as D
    elif mode == 'LR':
        from data.LRDataset import LRDataset as D
    elif mode == 'HRLANDMARK':
        from data.HRLandmarkDataset import HRLandmarkDataset as D
    elif mode == 'HRLANDMARKBLUR':
        from data.HRLandmarkBlurDataset import HRLandmarkBlurDataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    dataset = D(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset
