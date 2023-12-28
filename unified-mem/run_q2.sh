rm -f q2 q2*.txt
nvcc -o q2 q2.cu
nvprof --log-file q2_scenario_1.txt ./q2 scenario-1 
nvprof --log-file q2_scenario_2.txt ./q2 scenario-2 
nvprof --log-file q2_scenario_3.txt ./q2 scenario-3 

