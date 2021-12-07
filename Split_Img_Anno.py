# 将RotateVOC2007的Annotation和JPEGImages
# 文件夹分割成Train和Test。

import os
import shutil

# voc路径
root_path = '/home/wujian/data/RVOC2007/'
src_img_path = root_path + "JPEGImages/"
src_xml_path = root_path + "RRec_Annotations/"

# 文件名
trainval_path = root_path + "ImageSets/Main/trainval.txt" # trainval.txt
test_path = root_path + "ImageSets/Main/test.txt"         # test.txt

# 保存路径
save_train_img_path = root_path + "Train/AllImages/"
save_train_xml_path = root_path + "Train/Annotations/"
save_test_img_path = root_path + "Test/AllImages/"
save_test_xml_path = root_path + "Test/Annotations/"

# 保存trainval
with open(trainval_path,'r') as f:
    for ele in f.readlines():
        cur_name = ele.strip()
        shutil.copy(src_img_path + cur_name + '.jpg', save_train_img_path + cur_name + '.jpg')
        shutil.copy(src_xml_path + cur_name + '.xml', save_train_xml_path + cur_name + '.xml')
    print('Trainval Done!')

# 保存test
with open(test_path,'r') as f:
    for ele in f.readlines():
        cur_name = ele.strip()
        shutil.copy(src_img_path + cur_name + '.jpg', save_test_img_path + cur_name + '.jpg')
        shutil.copy(src_xml_path + cur_name + '.xml', save_test_xml_path + cur_name + '.xml')
    print('Test Done!')