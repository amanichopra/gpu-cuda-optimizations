rm -f q1 gmon.out q1*.txt 
g++ -o q1 q1.cpp -pg
./q1 1
gprof ./q1 gmon.out > q1_1.txt
./q1 5
gprof ./q1 gmon.out > q1_5.txt
./q1 10
gprof ./q1 gmon.out > q1_10.txt
./q1 50
gprof ./q1 gmon.out > q1_50.txt
./q1 100
gprof ./q1 gmon.out > q1_100.txt
