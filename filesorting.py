import pathlib
import shutil
import os

files = os.listdir('/Users/johnxie/Documents/Summer2021/archive/train')
fileList = os.listdir('/Users/johnxie/Documents/Summer2021/archive/mask')
files.sort()
fileList.sort()

for i in range(len(files)):
    print(files[i], fileList[i])
# 
# 

# print(root)
# print(location)





