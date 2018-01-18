# Weakly Supervised Learning for Findings Detection in Medical Images(HTC)

> 組名：rectangle

> 組員：[吳俊德], [郭勅君], [吳中群], [黃于真], [韓宏光]

## 1. Required Environment
### - OS version
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
  
- **Our model**

  (Automatically be downloaded by our script)

### - Module & Version Requirement
**Module Name**|**Version**
---|---
numpy|1.13.3
pandas|0.21.0
skimage|0.13.1
tensorflow|1.4.0
scipy|0.19.1
cv2(opencv-python)|3.3.0
skimage(python-skimage)|0.13.1

## 2. How to train
```
# [img_path]: Directory containing images(x-ray)
sh train.sh [img_path] # e.g. sh train.sh ./images/
```
## 3. How to Test
```
sh test.sh
```
