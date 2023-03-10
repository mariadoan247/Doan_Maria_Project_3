in=/home/oucspdn086/test_data/DNA_Files/GRCh38_test_rna.fna # starter input file name, can set like "make run_starter in=other.fna"

all: 
	make Exp1
	make Exp2


# duplicate this for other files
Exp1:
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_critical compute_average_TF_Exp1_critical.c -lm -std=c99
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_atomic compute_average_TF_Exp1_atomic.c -lm -std=c99
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_locks compute_average_TF_Exp1_locks.c -lm -std=c99


Exp2:
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp2_schedule compute_average_TF_Exp2_schedule.c -lm -std=c99


clean:
	$(RM) compute_average_TF_Exp1_critical
	$(RM) compute_average_TF_Exp1_atomic
	$(RM) compute_average_TF_Exp1_locks
	$(RM) compute_average_TF_Exp2_schedule

# duplicate this for other files
run_Exp1:
	./compute_average_TF_Exp1_critical $(in) /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_critical_50_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_critical_50_8th.csv 8
	./compute_average_TF_Exp1_atomic $(in) /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_atomic_50_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_atomic_50_8th.csv 8
	./compute_average_TF_Exp1_locks $(in) /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_locks_50_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_locks_50_8th.csv 8

# duplicate this for other files
run_Exp2:
	./compute_average_TF_Exp2_schedule $(in) /home/oucspdn086/Doan_Maria_Project_3/Problem_2/res_schedule_50_8th.csv /home/oucspdn086/Doan_Maria_Project_3/Problem_2/tim_schedule_50_8th.csv 8