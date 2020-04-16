import os
from collections import OrderedDict
from datetime import datetime
import json


from utils import util

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # export CUDA_VISIBLE_DEVICES
    if 'gpu_ids' in opt.keys():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
        import torch
        opt['use_gpu'] = torch.cuda.is_available()
    else:
        opt['use_gpu'] = False
        print('===> CPU mode is set (NOTE: GPU is recommended)')

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range
        
    # for network initialize
    if 'global' in opt['networks'].keys():
        opt['networks']['global']['scale'] = opt['scale']
        opt['networks']['local']['scale'] = opt['scale']
    else:
        opt['networks']['scale'] = opt['scale']
    network_opt = opt['networks']

    config_str = network_opt['which_model'].upper() + '_'
    if 'in_channels' in network_opt.keys(): config_str += 'in%d' % network_opt['in_channels']
    if 'num_features' in network_opt.keys(): config_str += 'f%d' % network_opt['num_features']
    config_str += '_x%d' % opt['scale']
    if 'name' in opt.keys():
        config_str +=  '_' + opt['name']

    if opt['is_train']:
        exp_path = os.path.join(opt['path']['root'], 'experiments', config_str)

        if opt['is_train'] and opt['solver']['pretrain'] == 'resume':
            if 'pretrained_path' not in list(opt['solver'].keys()): raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")
            exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
            if opt['solver']['pretrain'] == 'finetune': exp_path += '_finetune'

        exp_path = os.path.relpath(exp_path)

        path_opt = opt['path']
        path_opt['exp_root'] = exp_path
        path_opt['tb_logger_root'] = exp_path.replace('experiments', 'tb_logger')
        path_opt['epochs'] = os.path.join(exp_path, 'epochs')
        path_opt['visual'] = os.path.join(exp_path, 'visual')
        path_opt['records'] = os.path.join(exp_path, 'records')
    else:
        res_path = os.path.join(opt['path']['root'], 'results', config_str)
        res_path = os.path.relpath(res_path)
        path_opt = OrderedDict()
        path_opt['res_root'] = res_path

    opt['path'] = path_opt

    opt = dict_to_nonedict(opt)

    return opt


def save(opt):
    if 'exp_root' in opt['path'].keys():
        dump_dir = opt['path']['exp_root']
    else:
        dump_dir = opt['path']['res_root']

    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
