import pathlib
import shutil
import os

path = os.getcwd()
print(path)

files = os.listdir(path + '\\data\\train')
files.sort()

print(files)

for i in files:
    i = path + '\\data\\train\\' + i
    if "_mask" in i:
        temp = i.replace("train", "mask")
        print(i)
        print(temp)
        os.rename(i, temp)

# print(root)
# print(location)





