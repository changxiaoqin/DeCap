import os
from PIL import Image
import sys

dataset = sys.argv[0]
image_folder = './dataset/{}'.format(dataset)
file_names = os.listdir(image_folder)
print(file_names)

txt_file = './dataset/{}/cls_classes.txt'
print("begin")

with open(txt_file, 'w') as f:
    for file_name in file_names:
        label = file_name.replace("_", " ")
        f.write(file_name + '\n')
print("down")
