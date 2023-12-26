import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import SimpleITK as sitk
import os
import torch

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob.dim()==4:
        n, c, h, w = prob.size()
    if prob.dim()==5:
        n, c, h, w, t = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def prob_2_entropy_with_weight(prob_origin,class_origin,weighttumor=2.5,weightkidney=1):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob_origin.dim()==5:
        n, c, h, w, t = prob_origin.size()
    prob_origin = torch.squeeze(prob_origin, dim=0)
    class_map=np.expand_dims(class_origin, 0).repeat(3, axis=0)
    prob=prob_origin.clone().numpy()
    # cal entropy of only the tumor part------------------------
    if 2 in class_origin:
        prob[class_map != 2]=0
        prob = torch.from_numpy(prob).float()
        tumor_entropy=-torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
        tumor_entropy=np.sum(tumor_entropy.numpy())/np.sum(class_origin==2)/3
    else:
        tumor_entropy=0
    # cal entropy of only the kidney part-----------------------
    prob=prob_origin.clone().numpy()
    prob[class_map != 1]=0
    prob = torch.from_numpy(prob).float()
    kidney_entropy=-torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    kidney_entropy=np.sum(kidney_entropy.numpy())/np.sum(class_origin==1)/3

    return weighttumor*tumor_entropy+weightkidney*kidney_entropy

# # Test code
# case='case_0034000'
# arr = np.load('/Users/yeshuai/Documents/GraduateDesign/nnUnet/AdvEntCT/progress/modified_IntraDA/easy dataset/dui/'+case+'.npz')['softmax']
# img = sitk.ReadImage('/Users/yeshuai/Documents/GraduateDesign/nnUnet/AdvEntCT/progress/modified_IntraDA/easy dataset/dui/'+case+'.nii.gz')
# imgarr = sitk.GetArrayFromImage(img)
# slide = arr[np.newaxis, ...] #(1,3,z,w,h)
# slide = torch.from_numpy(slide).float()
# pred_trg_entropy = prob_2_entropy_with_weight(slide,imgarr)
# print(pred_trg_entropy)



# usage: Predict target domain dataset with best trained model to get the .npy file with origin prob(0to1) value after
#        With entropy value of each case, we sort and write it into a .txt file for latter divide.
# model_path='/seushare2/home/yangguanyu/ys/AdventCT/model/anew_a'

def origin_rank(weighted_entropy_rank,npz_path,postprocessd_ct_path,outtxt_path):
    def maybe_mkdir_p(directory):
        #        softmax. Calculate every .npy of each case to divide easy and hard case dataset finally.
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

    all_list=[]
    for f in os.listdir(npz_path):
        if f[-3:]=='npz': #case_0000000.npz
            print('processing ',f,flush=True)
            arr = np.load(npz_path+'/'+f)['softmax']
            slide=arr[:,...] # (3,z,w,h)
            img = sitk.ReadImage(postprocessd_ct_path + '/' + f[:-4] + '.nii.gz')
            imgarr = sitk.GetArrayFromImage(img)
            slide = slide[np.newaxis, ...]  # (1,3,z,w,h)

            slide = torch.from_numpy(slide).float()
            pred_trg_entropy = prob_2_entropy(slide)
            e=pred_trg_entropy.mean().item()
            all_list.append((f[:-4], e))
            print('\tcase:',f,'\tentropy:',e)


    print('------------------------------------------------')
    print(len(all_list),' cases ')
    print('\n\nsorting... ')
    entropy_list = sorted(all_list, key=lambda img: img[1])
    # for i in entropy_list:
    #     print(i,flush=True)

    maybe_mkdir_p(outtxt_path)
    with open(outtxt_path+'/no_tumor_entropy.txt','w+') as f:
        for item in entropy_list:
            # f.write('%s %f\n' % item)
            pass
    with open(outtxt_path + '/with_tumor_entropy.txt', 'w+') as f:
        for item in entropy_list:
            f.write('%s %f\n' % item)

if __name__ == "__main__":
    weighted_entropy_rank = False
    npz_path = '/seushare2/home/yangguanyu/ys/AdventCT/model_3d/m5/in_training'
    postprocessd_ct_path = '/seushare2/home/yangguanyu/ys/AdventCT/model_3d/m5/in_training/train_result/postprocess'
    # outtxt_path = '/seu_share/home/yangguanyu/ys/AdventCT/model_3d/m5/in_training/train_result/dataset/weighted_entroy_ranking'
    outtxt_path = '/seushare2/home/yangguanyu/ys/AdventCT/model_3d/m5/in_training/train_result/dataset/origin_intraDA_entroy_ranking'
    origin_rank(weighted_entropy_rank,npz_path,postprocessd_ct_path,outtxt_path)










