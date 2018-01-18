#!/bin/bash

echo "========================================================="
if [ ! -f vgg19.npy ]; then
		echo "[-] download vgg19.npy"
		echo ""
    curl ftp://140.112.107.150/vgg19.npy -o vgg19.npy

# $1 -> eg: images/
echo "[-] generate .npy files of img & label"

python3 handle.py $1

echo "[v] Files generated."

echo "Start Training..."

python3 train.py

