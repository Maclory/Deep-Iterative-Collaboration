import torch
import torch.nn as nn

class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.is_train = opt['is_train']
        self.use_chop = opt['use_chop']
        self.self_ensemble = opt['self_ensemble']
        self.use_cl = True if opt['use_cl'] else False

        # GPU verify
        self.use_gpu = torch.cuda.is_available() and 'gpu_ids' in opt.keys()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # experimental dirs
        self.exp_root = opt['path']['exp_root']
        self.checkpoint_dir = opt['path']['epochs']
        self.records_dir = opt['path']['records']
        self.visual_dir = opt['path']['visual']

        # log and vis scheme
        self.save_ckp_step = opt['solver']['save_ckp_step']
        self.save_vis_step = opt['solver']['save_vis_step']

        self.best_epoch = 0
        self.best_step = 0
        self.cur_epoch = 1
        self.step = 0
        self.best_pred = 0.0

    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def get_current_log(self):
        pass

    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load(self):
        pass

    def save_current_visual(self, epoch, iter):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n
