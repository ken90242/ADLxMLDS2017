# Weakly Supervised Learning for Findings Detection in Medical Images(HTC)

> 組名：rectangle

> 組員：[吳俊德], [郭勅君], [吳中群], [黃于真], [韓宏光]

## 1. Required Environment
#### - OS 
```Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-104-generic x86_64)```

### - Python Version
```Python 3.5.2```

### - Data Needed
- **Image files** list in **Data_Entry_2017_v2.csv**
  (You can download from ftp://140.112.107.150/DeepQ-Learning.zip)
- **npy files**(img feature & label vector)
  (Automatically be downloaded by our script)
- **Pretrain vgg19 original model**
  (Automatically be downloaded by our script)
- **Our vgg19 original model**
  (Automatically be downloaded by our script)

### Module & Version Requirement 

## 2. How to train
```
# [img_path]: Directory containing images(x-ray)
sh train.sh [img_path]
```
## 3. How to Test
```
sh test.sh
```
