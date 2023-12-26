import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import SimpleITK as sitk
import os
import torch
#read npz in outfold, add every entropy value with name_slidenum into a list
#sort with entropy value to pose easy and hard dataset, save the txt
from nnunet.training.network_training.nnUNetTrainerV2 import prob_2_entropy

model_path='/seu_share/home/yangguanyu/ys/AdventCT/model/anew_a'
npz_path='/seu_share/home/yangguanyu/ys/AdventCT/model/anew_a/predict_250'
outtxt_path='/seu_share/home/yangguanyu/ys/AdventCT/model/anew_a/predict_250/pseudo_label/07/'
# npz_path=model_path+'/predict'
# outtxt_path=model_path+'/pseudo_dataset/05/'
lamb=0.7

def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

entropy_list=[]
for f in os.listdir(npz_path):
    if f[-3:]=='npz':
        print('processing ',f,flush=True)
        arr = np.load(npz_path+'/'+f)['softmax']
        for z in range(arr.shape[1]):
            slide=arr[:,z,...] # (3,w,h)
            slide = slide[np.newaxis, ...]
            slide = torch.from_numpy(slide).float()
            pred_trg_entropy = prob_2_entropy(slide)
            e=pred_trg_entropy.mean().item()
            entropy_list.append((f[:-4].split('_')[-1]+'_'+str(z), e))
            print('\tz:',z,'\tentropy:',e)

print('\n\nsorting... ')
entropy_list = sorted(entropy_list, key=lambda img: img[1])
for i in entropy_list:
    print(i,flush=True)
copy_list = entropy_list.copy()
entropy_rank = [item[0] for item in entropy_list]

easy_split = entropy_rank[ : int(len(entropy_rank) * lamb)]
hard_split = entropy_rank[int(len(entropy_rank)* lamb): ]

maybe_mkdir_p(outtxt_path)
with open(outtxt_path+'easy_split.txt','w+') as f:
    for item in easy_split:
        f.write('%s\n' % item)

with open(outtxt_path+'hard_split.txt','w+') as f:
    for item in hard_split:
        f.write('%s\n' % item)








