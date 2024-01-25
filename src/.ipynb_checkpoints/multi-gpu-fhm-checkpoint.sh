#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=10           # Number of CPU to request for the job
#SBATCH --mem=20GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=hatred.out            # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --constraint=a40
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --account=jiangjingresearch   # The account you've been assigned (normally student)
#SBATCH --qos=20231231-ruicao.2020     # What is the QOS assigned to you? Check with myinfo command
#SBATCH --job-name=tune-llama-cap    # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
#module purge
#module load Anaconda3/2022.05

# Create a virtual environment can be commented off if you already have a virtual environment
# conda create -n myenvnamehere

# Do not remove this line even if you have executed conda init
#eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
#conda activate myenvname

# If you require any packages, install it before the srun job submission.
#conda install pytorch torchvision torchaudio -c pytorch

# Submit your job to the cluster
#srun --partition=researchshort --gres=gpu:2 python tuning-fhm.py --batch_size 64 --micro_batch_size 4 --num_epochs 3 --learning_rate 2e-5 --save_step 100 --eval_step 50
#srun --partition=researchshort --gres=gpu:2 python tuning-fhm.py --batch_size 64 --micro_batch_size 4 --num_epochs 3 --learning_rate 5e-5 --save_step 100 --eval_step 50
#srun --partition=researchshort --gres=gpu:2 python tuning-fhm.py --batch_size 64 --micro_batch_size 4 --num_epochs 3 --learning_rate 1e-4 --save_step 100 --eval_step 50
srun --partition=researchshort --gres=gpu:2 python main.py --batch_size 8 --micro_batch_size 4 --num_epochs 2 --learning_rate 5e-4 --save_step 100 --gen_step 10000 --DATASET 'hatred'
