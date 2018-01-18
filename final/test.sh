#!/bin/bash

echo "========================================================="
if [ ! -f test-save_24.npy ]; then
		echo "[-] download test-save_24.npy"
		echo ""
    curl ftp://140.112.107.150/test-save_24.npy -o test-save_24.npy

echo "Start Testing..."

python3 judge.py