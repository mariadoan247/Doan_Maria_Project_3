in=in.fna # starter input file name, can set like "make run_starter in=other.fna"

all: 
	make Exp1
	make Exp2


# duplicate this for other files
make Exp1:
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_critical compute_average_TF_Exp1_critical.c -lm -std=c99
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_atomic compute_average_TF_Exp1_atomic.c -lm -std=c99
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp1_locks compute_average_TF_Exp1_locks.c -lm -std=c99


make Exp2:
	gcc -fopenmp -g -Wall -o compute_average_TF_Exp2_schedule compute_average_TF_Exp2_schedule.c -lm -std=c99


clean:
	$(RM) compute_average_TF_Exp1_critical
	$(RM) compute_average_TF_Exp1_atomic
	$(RM) compute_average_TF_Exp1_locks
	$(RM) compute_average_TF_Exp2_schedule

