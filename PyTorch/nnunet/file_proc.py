import os
import shutil
import random


source_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/kits_data/alldata/ct')
target_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/AdventCT/dataset/mixed/tr')

if os.path.exists(source_path):
    files=os.listdir(source_path)
    random.shuffle(files)
    for file in files[:80]:
        # case_00113_0000.nii.gz
        src_file = os.path.join(source_path, file) 
        target_file = os.path.join(target_path, file[:10]+"00_0000.nii.gz")

        shutil.copy(src_file, target_file)
        print(src_file)

print('target ct copy files finished!')


source_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/ours_data/nnUNet_raw_data_base/nnUNet_raw_data/Task102_Ours/imagesTr')
target_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/AdventCT/dataset/mixed/tr')

if os.path.exists(source_path):
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # case_0657737_0000.nii.gz
            src_file = os.path.join(root, file)  # getrid of 0000
            target_file = os.path.join(target_path, file[:12]+"_0000.nii.gz")

            shutil.copy(src_file, target_file)
            print(src_file)

print('source tr copy files finished!')

source_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/ours_data/nnUNet_raw_data_base/nnUNet_raw_data/Task102_Ours/labelsTr')
target_path = os.path.abspath(r'/seu_share/home/yangguanyu/ys/AdventCT/dataset/mixed/gt')

if os.path.exists(source_path):
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # case_0657737.nii.gz
            src_file = os.path.join(root, file) 
            target_file = os.path.join(target_path, file) 

            shutil.copy(src_file, target_file)
            print(src_file)

print('source gt copy files finished!')


