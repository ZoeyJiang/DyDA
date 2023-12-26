import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import SimpleITK as sitk
import os
import torch

def maybe_mkdir_p(directory):
    #        softmax. Calculate every .npy of each case to divide easy and hard case dataset finally.
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob.dim() == 4:
        n, c, h, w = prob.size()
    if prob.dim() == 5:
        n, c, h, w, t = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def prob_2_entropy_with_weight_kits(prob_origin, class_origin, weighttumor=1, weightkidney=1):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob_origin.dim() == 5:
        n, c, h, w, t = prob_origin.size()
    prob_origin = torch.squeeze(prob_origin, dim=0)
    class_map=np.expand_dims(class_origin, 0).repeat(3, axis=0)
    prob=prob_origin.clone().numpy()
    # cal entropy of only the tumor part------------------------
    if 2 in class_origin:
        prob[class_map != 2] = 0
        prob = torch.from_numpy(prob).float()
        tumor_entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
        tumor_entropy = np.sum(tumor_entropy.numpy())/np.sum(class_origin==2)/3
    else:
        tumor_entropy = 0
    # cal entropy of only the kidney part-----------------------
    prob = prob_origin.clone().numpy()
    prob[class_map != 1] = 0
    prob = torch.from_numpy(prob).float()
    kidney_entropy=-torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    kidney_entropy=np.sum(kidney_entropy.numpy())/np.sum(class_origin == 1)/3

    return weighttumor*tumor_entropy+weightkidney*kidney_entropy

def prob_2_entropy_with_weight_acdc(prob_origin,class_origin,weight_RV=1,weight_LMB=3, weight_LV=1):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob_origin.dim() == 4:
        n, c, h, w = prob_origin.size()
    if prob_origin.dim() == 5:
        n, c, h, w, t = prob_origin.size()
    prob_origin = torch.squeeze(prob_origin, dim=0)
    class_map = np.expand_dims(class_origin, 0).repeat(4, axis=0)
    class_map = np.resize(class_map, prob_origin.size())

    # cal entropy of only RV part--------------
    prob = prob_origin.clone().numpy()
    prob[class_map != 1] = 0
    prob = torch.from_numpy(prob).float()
    RV_entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    RV_entropy = np.sum(RV_entropy.numpy())/np.sum(class_origin == 1)/4

    # cal entropy of only the LMB part------------------------
    prob = prob_origin.clone().numpy()
    prob[class_map != 2] = 0
    prob = torch.from_numpy(prob).float()
    LMB_entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    LMB_entropy = np.sum(LMB_entropy.numpy()) / np.sum(class_origin == 2) / 4

    # cal entropy of only the LV part-----------------------
    if 3 in class_origin:
        prob = prob_origin.clone().numpy()
        prob[class_map != 3] = 0
        prob = torch.from_numpy(prob).float()
        LV_entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
        LV_entropy = np.sum(LV_entropy.numpy())/np.sum(class_origin == 3)/4
    else:
        LV_entropy = 0

    print('Entropy: ', RV_entropy, LMB_entropy, LV_entropy)
    return weight_RV*RV_entropy + weight_LMB*LMB_entropy + weight_LV*LV_entropy


def rank_jsph_kits(weighted_entropy_rank, npz_path, postprocessd_ct_path, outtxt_path):
    no_tumor_entropy_list=[]
    with_tumor_entropy_list=[]
    for f in os.listdir(npz_path):
        if f[-3:]=='npz': #case_0000000.npz
            print('processing ',f,flush=True)
            arr = np.load(npz_path+'/'+f)['softmax']
            slide=arr[:,...] # (3,z,w,h)
            img = sitk.ReadImage(postprocessd_ct_path + '/' + f[:-4] + '.nii.gz')
            imgarr = sitk.GetArrayFromImage(img)
            slide = slide[np.newaxis, ...]  # (1,3,z,w,h)
            if weighted_entropy_rank:
                slide = torch.from_numpy(slide).float()
                e = prob_2_entropy_with_weight_kits(slide,imgarr)
                if 2 in imgarr:
                    with_tumor_entropy_list.append((f[:-4], e))
                else:
                    no_tumor_entropy_list.append((f[:-4], e))
                print('\tcase:',f,'\tweighted entropy:',e)
            else:
                slide = torch.from_numpy(slide).float()
                pred_trg_entropy = prob_2_entropy(slide)
                e=pred_trg_entropy.mean().item()
                if 2 in imgarr:
                    with_tumor_entropy_list.append((f[:-4], e))
                else:
                    no_tumor_entropy_list.append((f[:-4], e))
                print('\tcase:',f,'\tentropy:',e)


    print('------------------------------------------------')
    print('no tumor ',len(no_tumor_entropy_list),' cases, with tumor ',len(with_tumor_entropy_list),' cases.')
    print('\n\nsorting... ')
    no_tumor_entropy_list = sorted(no_tumor_entropy_list, key=lambda img: img[1])
    with_tumor_entropy_list = sorted(with_tumor_entropy_list, key=lambda img: img[1])
    # for i in entropy_list:
    #     print(i,flush=True)

    maybe_mkdir_p(outtxt_path)
    with open(outtxt_path+'/no_tumor_entropy.txt','w+') as f:
        for item in no_tumor_entropy_list:
            f.write('%s %f\n' % item)
    with open(outtxt_path + '/with_tumor_entropy.txt', 'w+') as f:
        for item in with_tumor_entropy_list:
            f.write('%s %f\n' % item)


def rank_acdc_myo_emidec(weighted_entropy_rank, npz_path, postprocessd_ct_path, outtxt_path):
    no_LV_entropy_list = []
    with_LV_entropy_list = []

    for f in os.listdir(npz_path):
        if f[-3:] == 'npz': #case_0000000.npz
            arr = np.load(npz_path+'/'+f)['softmax']
            slide = arr[np.newaxis:,...] # (3,z,w,h)
            slide = torch.from_numpy(slide).float()

            img = sitk.ReadImage(postprocessd_ct_path + '/' + f[:-4] + '.nii.gz')
            imgarr = sitk.GetArrayFromImage(img)

            if weighted_entropy_rank:
                e = prob_2_entropy_with_weight_acdc(slide, imgarr)
                if 3 in imgarr:
                    with_LV_entropy_list.append((f[:-4], e))
                else:
                    no_LV_entropy_list.append((f[:-4], e))
                print('\tcase:', f, '\tweighted entropy:', e)
            else:
                pred_trg_entropy = prob_2_entropy(slide)
                e = pred_trg_entropy.mean().item()
                if 3 in imgarr:
                    with_LV_entropy_list.append((f[:-4], e))
                else:
                    no_LV_entropy_list.append((f[:-4], e))
                print('\tcase:', f, '\tentropy:', e)

    print('------------------------------------------------')
    print('no LV ',len(no_LV_entropy_list), ' cases, with LV ', len(with_LV_entropy_list), ' cases.')
    no_tumor_entropy_list = sorted(no_LV_entropy_list, key=lambda img: img[1])
    with_tumor_entropy_list = sorted(with_LV_entropy_list, key=lambda img: img[1])
    maybe_mkdir_p(outtxt_path)
    with open(outtxt_path+'/no_LV_entropy_list.txt', 'w+') as f:
        for item in no_tumor_entropy_list:
            f.write('%s %f\n' % item)
    with open(outtxt_path + '/with_LV_entropy_list.txt', 'w+') as f:
        for item in with_tumor_entropy_list:
            f.write('%s %f\n' % item)
    print('\n\nsorting done... ')


dir_path = '/media/F/jzy/DyDE/dataset/results/Task011/lambda07/DyDA_Min/dy_predict_result_0'
case = 'Case_N006'

npz_arr = np.load(os.path.join(dir_path, case+'.npz'))
arr = npz_arr['softmax']
slide = arr[np.newaxis:, ...]

img = sitk.ReadImage(os.path.join(dir_path, case+'.nii.gz'))
img_arr = sitk.GetArrayFromImage(img)

print(case, prob_2_entropy_with_weight_acdc(slide, img_arr))



# case_list = []
# for file in os.listdir(dir_path):
#     if file.endswith('.nii.gz'):
#         case_list.append(file.split('.nii.gz')[0])
#
# for case in case_list:
#     arr = np.load(dir_path+case+'.npz')['softmax']
#     img = sitk.ReadImage(dir_path+case+'.nii.gz')
#     imgarr = sitk.GetArrayFromImage(img)
#     slide = arr[np.newaxis:, ...]
#     slide = torch.from_numpy(slide).float()
#     print(case, prob_2_entropy_with_weight_acdc(slide, imgarr), '-----\n')


 # test results-------------------