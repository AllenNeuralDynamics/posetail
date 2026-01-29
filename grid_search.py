import argparse 
import itertools 
import os 
import random
import shutil
import string
import subprocess
import toml

import numpy as np 
import pandas as pd 

from posetail.train_utils import *

''' 
example script submission. only use the auto-submit flag to submit each job to a slurm cluster

python grid_search.py --config-path configs/config_default_3d.toml
python grid_search.py --config-path configs/config_default_3d.toml --auto-submit
'''

# list of parameter combinations to test - each must be the same length
PARAM_DICT = {'training.losses.pixel_thresh': [3, 6]}

def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default = './configs/config_default.toml')
    parser.add_argument('--auto-submit', action = 'store_true', help = 'auto submits job to slurm')
    parser.add_argument('--base-dir', help = 'base repo path, used for auto-submit')
    parser.add_argument('--tmp-dir', help = 'tmp dir path, used for auto-submit')
    # base_dir = '/home/katie.rupp/posetail/'
    # tmp_dir = '/allen/aind/scratch/katie.rupp/tmp'

    args = parser.parse_args()

    return args

def save_config(config, outpath):
    '''
    save config to the provided outpath
    '''
    with open(outpath, 'w') as toml_file:
        toml.dump(config, toml_file)

def safe_make(path): 

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def generate_uuid(n = 24):
    ''' 
    generates a unique id of the given length
    '''
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    uuid = [random.choice(alphabet) for _ in range(n)]
    uuid = ''.join(uuid)

    return uuid

def get_combinations(param_dict): 
    ''' 
    gets a list of dictionaries of all possible parameter combinations
    '''
    combinations = list(itertools.product(*param_dict.values()))
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

    return default_config


def main(args): 

    default_config = load_config(args.config_path)
    prefix =  os.path.dirname(args.config_path)

    if PARAM_DICT is not None:

        # update default config with new params
        config_paths = []
        param_dicts = get_combinations_simple(PARAM_DICT)

        for i, param_dict in enumerate(param_dicts): 

            new_config = update_config(default_config, param_dict)
            outpath = os.path.join(prefix, f'config{i}.toml')
            save_config(new_config, outpath)
            config_paths.append(outpath)
            print(f'creating config {outpath}')

    else: 
        config_paths = [args.config_path]

    return config_paths


if __name__ == '__main__':

    args = parse_args()
    config_paths = main(args)

    if args.auto_submit:

        # ensure we are in the main codebase
        os.chdir(args.base_dir)
        print(f'running from {os.getcwd()}')

        # run the submission script for each config
        for config_path in config_paths:

            print(f'submitting job with config {config_path}')
            result = subprocess.run(f'bsub {config_path} < train.sh', shell = True, check = True)
