#!/bin/sh

# ./mulIPU
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./graph_report"}' PVTI_OPTIONS='{"enable":"true", "directory": "./system_report"}' 
# ./mulIPUtc -g 3 -n 1 -v
# ./pr_spmv -g 14 -n 1 -v -i 30
# gdb -ex=r --args 

POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./graph_report4knaive"}' PVTI_OPTIONS='{"enable":"true", "directory": "./system_report4knaive"}' ./tc -g $1 -n 1 -v
# ./tc -g $1 -n 1 -v
