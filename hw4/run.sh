#!/bin/bash

if [ ! -f model.data-00000-of-00001 ]; then
    curl ftp://140.112.107.150/r06725053/model.data-00000-of-00001 -o model.data-00000-of-00001
    echo "download model.data-00000-of-00001"
fi
echo "[v] checked model.data-00000-of-00001"
if [ ! -f model.index ]; then
    curl ftp://140.112.107.150/r06725053/model.index -o model.index
    echo "[-] download model.index"
fi
echo "[v] checked model.index"
if [ ! -f model.meta ]; then
    curl ftp://140.112.107.150/r06725053/model.meta -o model.meta
    echo "[-] download model.meta"
fi
echo "[v] checked model.meta"
if [ ! -f Data/skipthoughts/bi_skip.npz ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/bi_skip.npz -o Data/skipthoughts/bi_skip.npz
    echo "[-] download bi_skip.npz"
fi
echo "[v] checked bi_skip.npz"
if [ ! -f Data/skipthoughts/bi_skip.npz.pkl ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/bi_skip.npz.pkl -o Data/skipthoughts/bi_skip.npz.pkl
    echo "[-] download bi_skip.npz.pkl"
fi
echo "[v] checked bi_skip.npz.pkl"
if [ ! -f Data/skipthoughts/btable.npy ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/btable.npy -o Data/skipthoughts/btable.npy
    echo "[-] download btable.npy"
fi
echo "[v] checked btable.npy"
if [ ! -f Data/skipthoughts/dictionary.txt ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/dictionary.txt -o Data/skipthoughts/dictionary.txt
    echo "[-] download dictionary.txt"
fi
echo "[v] checked dictionary.txt"
if [ ! -f Data/skipthoughts/uni_skip.npz ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/uni_skip.npz -o Data/skipthoughts/uni_skip.npz
    echo "[-] download uni_skip.npz"
fi
echo "[v] checked uni_skip.npz"
if [ ! -f Data/skipthoughts/uni_skip.npz.pkl ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/uni_skip.npz.pkl -o Data/skipthoughts/uni_skip.npz.pkl
    echo "[-] download uni_skip.npz.pkl"
fi
echo "[v] checked uni_skip.npz.pkl"
if [ ! -f Data/skipthoughts/utable.npy ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/utable.npy -o Data/skipthoughts/utable.npy
    echo "[-] download utable.npy"
fi
echo "[v] checked utable.npy"
python3 generate.py $1