#!/bin/bash
#BSUB -J tribig 					# job name
#BSUB -W 00:10 						# walltime in HH:MM or MMMM
#BSUB -n 96 							# num CPU slots
#BSUB -M 64000 						# memory limit in MB
#BSUB -e /groups/karashchuk/home/karashchukl/logs/posetail/%J.err 	# error file
#BSUB -o /groups/karashchuk/home/karashchukl/logs/posetail/%J.out 	# out file
#BSUB -R "span[hosts=1]" 			# number of nodes spanned by gpus
#BSUB -q gpu_h200_parallel   		# type of gpu
#BSUB -gpu "num=8" 					# number of gpus

# define model params
CONFIG_PATH=${1:-"configs/config_default_3d.toml"}
NUM_GPUS=${2:-8}
PRECISION=${3:-"32"}
STRATEGY=${4:-"ddp"}
echo "using config $CONFIG_PATH"
echo "pwd: $PWD"

cd /groups/karashchuk/home/karashchukl/projects/posetail

source setup_env_vars.sh

# gpu specs
nvidia-smi

# pixi environment 
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
# pixi install 
# echo "pixi env setup complete"

# wandb login 
# wandb login $WANDB

# run training script 
echo "starting training..."
pixi run python train.py --config-path "${CONFIG_PATH}" --num_nodes ${NUM_GPUS} --precision ${PRECISION} --strategy ${STRATEGY} --devices 0 1 2 3 4 5 6 7
echo "done!"
