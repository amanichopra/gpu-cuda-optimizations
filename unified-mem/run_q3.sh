rm -f q3 q3*.txt
nvcc -o q3 q3.cu
nvprof --log-file q3_scenario_1.txt ./q3 scenario-1 
nvprof --log-file q3_scenario_2.txt ./q3 scenario-2 
nvprof --log-file q3_scenario_3.txt ./q3 scenario-3 
