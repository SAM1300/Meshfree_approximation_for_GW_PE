# !/bin/bash

taskset -c 32-63 python3 full_pe_code.py 0
taskset -c 32-63 python3 full_pe_code.py 1
taskset -c 32-63 python3 full_pe_code.py 2
taskset -c 32-63 python3 full_pe_code.py 3
taskset -c 32-63 python3 full_pe_code.py 4
taskset -c 32-63 python3 full_pe_code.py 5
taskset -c 32-63 python3 full_pe_code.py 6
taskset -c 32-63 python3 full_pe_code.py 7
taskset -c 32-63 python3 full_pe_code.py 8
taskset -c 32-63 python3 full_pe_code.py 9
taskset -c 32-63 python3 full_pe_code.py 10