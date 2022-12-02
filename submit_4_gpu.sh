#!/bin/bash
#SBATCH --job-name="tensorflow_mirroredstrategy"
#SBATCH --output="tensorflow_mirroredstrategy_%j.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cores-per-socket=20
#SBATCH --threads-per-core=4
#SBATCH --sockets-per-node=1
#SBATCH --mem-per-cpu=1200
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:4
#SBATCH --time=2:00:00

module load conda_base

conda activate shirui_env

python ./TCN_distributed.py --epochs 20 --batch_size 64 --dataPath '/home/shirui/student_consulting/seid_distributed/'

