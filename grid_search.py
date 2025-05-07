import argparse 
import itertools 
import os 

import numpy as np 
import pandas as pd 

from train_utils import *


def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default = './configs/config_default.toml')
    
    args = parser.parse_args()

    return args

def save_config(config, outpath):
    '''
    save config to the provided outpath
    '''
    with open(outpath, 'w') as toml_file:
        toml.dump(config, toml_file)

def get_combinations(param_dict): 
    ''' 
    gets a list of dictionaries of all possible parameter combinations
    '''
    get_combinations = list(itertools.product(*param_dict.values()))
    param_dicts = [dict(zip(param_dict.keys(), c)) for c in combinations]
    print(len(param_dicts))

    return param_dicts

def get_combinations_simple(param_dict): 

    param_dicts = []
    keys = list(param_dict.keys())
    combinations = len(param_dict[keys[0]])
    
    for i in range(combinations):
        params = {k: param_dict[k][i] for k in keys}
        param_dicts.append(params)

    return param_dicts

def update_config(default_config, param_dict):
    ''' 
    given a default configuration file, updates 
    the parameters with the new parameters provided
    in param dict
    '''
    for ks, v in param_dict.items(): 

        d = default_config
        keys = ks.split('.')

        for k in keys[:-1]: 
            d = d[k]
        
        d[keys[-1]] = v

    return default_confi


def main(args): 

    default_config = load_config(args.config_path)

    # list of parameter combinations to test 
    param_dict = {'dataset.batch_size': [1, 1, 1, 1, 1, 1], 
                'dataset.train.max_res': [256, 256, 256, 256, 256, 256], 
                'training.use_scheduler': [True, True, True, False, False, False],
                'training.optimizer.learning_rate': [0.0001, 0.0001, 0.005, 0.000001, 0.000001, 0.0000005], 
                'training.optimizer.weight_decay': [0.00001, 0.000001, 0.00001, 0.00001, 0.000001, 0.000005]}

    # update default config with new params
    config_paths = []
    param_dicts = get_combinations_simple(param_dict)

    for i, param_dict in enumerate(param_dicts):     
        new_config = update_config(default_config, param_dict)
        outpath = os.path.join(config_path, f'config{i}.toml')
        save_config(new_config, outpath)
        config_paths.append(outpath)


if __name__ == '__main__':

    args = parse_args()
    main(args)

    # !cd /home/katie.rupp/posetail
    # for config_path in config_paths: 
    #     !sbatch train.sh {config_path}