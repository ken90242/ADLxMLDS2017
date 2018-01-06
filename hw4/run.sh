#!/bin/bash
echo "========================================================="
if [ ! -f model.data-00000-of-00001 ]; then
		echo "[-] download model.data-00000-of-00001"
		echo ""
    curl ftp://140.112.107.150/r06725053/model.data-00000-of-00001 -o model.data-00000-of-00001
    
fi
echo ""
echo -e "[v] checked model.data-00000-of-00001"
echo "========================================================="
if [ ! -f model.index ]; then
		echo "[-] download model.index"
		echo ""
    curl ftp://140.112.107.150/r06725053/model.index -o model.index
fi
echo ""
echo -e "[v] checked model.index"
echo "========================================================="
if [ ! -f model.meta ]; then
    echo "[-] download model.meta"
    echo ""
    curl ftp://140.112.107.150/r06725053/model.meta -o model.meta
fi
echo ""
echo -e "[v] checked model.meta"
echo "========================================================="
if [ ! -f Data/skipthoughts/bi_skip.npz ]; then
    echo "[-] download bi_skip.npz"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/bi_skip.npz -o Data/skipthoughts/bi_skip.npz
fi
echo ""
echo -e "[v] checked bi_skip.npz"
echo "========================================================="
if [ ! -f Data/skipthoughts/bi_skip.npz.pkl ]; then
    echo "[-] download bi_skip.npz.pkl"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/bi_skip.npz.pkl -o Data/skipthoughts/bi_skip.npz.pkl
fi
echo ""
echo -e "[v] checked bi_skip.npz.pkl"
echo "========================================================="
if [ ! -f Data/skipthoughts/btable.npy ]; then
    echo "[-] download btable.npy"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/btable.npy -o Data/skipthoughts/btable.npy
fi
echo ""
echo -e "[v] checked btable.npy"
echo "========================================================="
if [ ! -f Data/skipthoughts/dictionary.txt ]; then
    echo "[-] download dictionary.txt"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/dictionary.txt -o Data/skipthoughts/dictionary.txt
fi
echo ""
echo -e "[v] checked dictionary.txt"
echo "========================================================="
if [ ! -f Data/skipthoughts/uni_skip.npz ]; then
    echo "[-] download uni_skip.npz"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/uni_skip.npz -o Data/skipthoughts/uni_skip.npz
fi
echo ""
echo -e "[v] checked uni_skip.npz"
echo "========================================================="
if [ ! -f Data/skipthoughts/uni_skip.npz.pkl ]; then
    echo "[-] download uni_skip.npz.pkl"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/uni_skip.npz.pkl -o Data/skipthoughts/uni_skip.npz.pkl
fi
echo ""
echo -e "[v] checked uni_skip.npz.pkl"
echo "========================================================="
if [ ! -f Data/skipthoughts/utable.npy ]; then
    echo "[-] download utable.npy"
    echo ""
    curl ftp://140.112.107.150/r06725053/skipthoughts/utable.npy -o Data/skipthoughts/utable.npy
fi
echo ""
echo -e "[v] checked utable.npy"
echo "========================================================="

python3 generate.py $1
