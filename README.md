# tc-ipu
Implementation of triangle counting for Graphcore IPU

Compile:

    g++ --std=c++17 -O3 -Wall -fopenmp -w -I ~/eigen/ tc_final.cc -lpoplar -lpopops -lpoplin -lpoputil -o tc

Run:

    sbatch --ipus=8 --output=report.o investigate.sh 12
