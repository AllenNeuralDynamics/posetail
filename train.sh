#!/bin/bash
#SBATCH --job-name=posetail                                     # job name
#SBATCH --mail-type=END,FAIL                                    # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=katie.rupp@alleninstitute.org               # where to send mail
#SBATCH -N 1                                                    # number of nodes requested
#SBATCH --mem=64G                                               # cpu memory 
#SBATCH --ntasks-per-node=1                                     # number of tasks to run on each node
#SBATCH --time=20:00:00                                          # time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                                       # gpu(s) requested
#SBATCH --output=/allen/aind/scratch/katie.rupp/slurm/%j.log    # standard output and error log
#SBATCH --partition aind                                        # partition used for processing
#SBATCH --tmp=30G                                               # request the amount of space your jobs needs on /scratch/fast
 
# variable definitions
CONDA_PATH="/allen/aind/scratch/katie.rupp/miniforge3"
CONDA_ENV_NAME="posetail118"
TEMP_DIR="/scratch/fast"
DATA_DIR="/allen/aind/scratch/katie.rupp/data/rat7m"

CONFIG_PATH=${1:-"configs/config_hpc_2d.toml"} 
echo "using config $CONFIG_PATH"
echo "pwd: $PWD"

# gpu specs
nvidia-smi
 
# activate conda environment
echo "activating conda env..."
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate $CONDA_PATH/envs/$CONDA_ENV_NAME
echo -e "$CONDA_DEFAULT_ENV env activated\n"

# move data to storage node for fast access
echo "copying data..."
mkdir -p $TEMP_DIR
cp -r $DATA_DIR $TEMP_DIR/
echo -e "data copied to ${TEMP_DIR}\n"

# log into wandb using api key
wandb login $WANDB

# run training script 
echo "starting training..."
python train_rat7m.py --config-path "${CONFIG_PATH}"
echo "done!"

# clean up the submission dir
echo "cleaning up submission dir..."
rm -rf "$SLURM_SUBMIT_DIR"
echo "done!"