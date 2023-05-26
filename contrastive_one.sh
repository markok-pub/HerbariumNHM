#!/bin/sh
#SBATCH -J herbarium_contr_1
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:1
python3 ./ntoebooks/contrastive_on_server.py