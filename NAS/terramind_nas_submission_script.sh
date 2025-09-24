#!/usr/bin/sh
#######################################################################
##
## This is a bash script used by qsub to submit a PBS job on NAS to the
## Milan a100 queue
##
#######################################################################

## Syntax:
## double # denotes a comment
## single # followed by PBS denotes a PBS configuration
## No # denotes a command to be run as part of the job

## Set the shell to use
#PBS -S /usr/bin/sh

## Set the name of the job
#PBS -N terramind-run1

## Note: model can be sky_gpu or cas_gpu or mil_a100
## for mil_a100, run on cfe and use queue: gpu_normal@pbspl4 for production
## and queue: gpu_devel@pbspl4 for development
## note: ncpus should be no less than the number of dataloader workers

## Define the resources for the job
#PBS -q gpu_devel@pbspl4
#PBS -l select=1:model=mil_a100:ncpus=16:ngpus=1:mem=120GB
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00

### Output files
#PBS -o terramind.out
#PBS -e terramind.err

## PBS will email when the job is aborted, begun, ended.
#PBS -M ffwilliams2@alaska.edu 
#PBS -m abe

## Load module and activate conda environment
module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
export CONDA_INSTALL=/nobackup/ffwillia/conda/envs
export CONDA_PKGS=/nobackup/ffwillia/conda/pkgs
conda config --add envs_dirs $CONDA_INSTALL
conda config --add pkgs_dirs $CONDA_PKGS
source activate terramind

export PYTHONPATH=$PWD
export NODE_RANK=$2
export RDZV_HOST=$(hostname)
export RDZV_PORT=19410
export WANDB_API_KEY=d2f6aaed7fc81a840c06d6b560d6d541be7945ee
export WANDB_DIR=$PWD/logs/

## Actual code to be run
echo "Hello World"
terratorch fit -c config.yml

## Cleanup
conda deactivate
