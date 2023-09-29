# tc-ipu
Implementation of triangle counting for Graphcore IPU

Compile:

    g++ --std=c++17 -O3 -Wall -fopenmp -w tc_final.cc -lpoplar -lpopops -lpoplin -lpoputil -o tc

Run triangle counting on a Kronecker graph of size 2^12 on 8 IPUs:

    sbatch --ipus=8 --output=report.o investigate.sh 12
