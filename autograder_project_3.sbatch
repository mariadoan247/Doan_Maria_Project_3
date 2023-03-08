#!/bin/bash
# Choose the partition that you will use.
# You can choose: oucspdn_cpu OR debug_5min OR debug OR others 
#
#SBATCH --partition=oucspdn_cpu
#
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#
#SBATCH --time=00:15:00
#SBATCH --job-name=auto
#
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=mariadoan247@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --chdir=/home/oucspdn086/Doan_Maria_Project_3/
#SBATCH --output=/home/oucspdn086/Doan_Maria_Project_3/autograding_%J_stdout.txt
#SBATCH --error=/home/oucspdn086/Doan_Maria_Project_3/autograding_%J_stderr.txt

#################################################

module purge

module load SciPy-bundle/2020.03-foss-2020a-Python-3.8.2

# To run all the problems and get all results
python3 -u autograder_project_3.py

# To run just problem 2's autograder
# python3 -u autograder_problem_3_2.py

# To run only Experiment 2 for Problem 2
# cd Problem_2
# make clean
# make Exp2
# ./compute_average_TF_Exp2_schedule /home/oucspdn086/test_data/DNA_Files/GRCh38_50_rna.fna /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_schedule_50_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_schedule_50_8th.csv 8
# ./compute_average_TF_Exp2_schedule /home/oucspdn086/test_data/DNA_Files/GRCh38_reduced_rna.fna /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_schedule_reduced_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_schedule_reduced_8th.csv 8
# ./compute_average_TF_Exp2_schedule /home/oucspdn086/test_data/DNA_Files/GRCh38_latest_rna.fna /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_schedule_latest_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_schedule_latest_8th.csv 8

# To run just problem 3's autograder
# python3 -u autograder_problem_3_3.py

# To run just problem 4's autograder
# python3 -u autograder_problem_3_4.py