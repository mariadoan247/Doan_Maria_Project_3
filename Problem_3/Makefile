in=/home/oucspdn086/test_data/DNA_Files/GRCh38_test_rna.fna # baseline input file name, can set like "make run_baseline in=other.fna"

all: 
	make Exp1
	make Exp2


# duplicate this for other files
Exp1: compute_median_TF_Exp1_baseline.c
	gcc -fopenmp -g -Wall -o compute_median_TF_Exp1_baseline compute_median_TF_Exp1_baseline.c -lm -std=c99


Exp2: compute_median_TF_Exp2_mapreduce.c
	gcc -fopenmp -g -Wall -o compute_median_TF_Exp2_mapreduce compute_median_TF_Exp2_mapreduce.c -lm -std=c99
	

clean:
	$(RM) compute_median_TF_Exp1_baseline
	$(RM) compute_median_TF_Exp2_mapreduce


# Below are commands to help you run your program easily.
# You will need to create more entries for your different files, such as for critical and locks.
run: 
	make run_baseline
	make run_mapreduce


# duplicate this for other files
run_baseline:
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_1th.csv TIME_baseline_1th.csv 1
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_2th.csv TIME_baseline_2th.csv 2
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_4th.csv TIME_baseline_4th.csv 4
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_8th.csv TIME_baseline_8th.csv 8

# duplicate this for other files
run_mapreduce:
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_1th.csv TIME_mapreduce_1th.csv 1
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_2th.csv TIME_mapreduce_2th.csv 2
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_4th.csv TIME_mapreduce_4th.csv 4
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_8th.csv TIME_mapreduce_8th.csv 8