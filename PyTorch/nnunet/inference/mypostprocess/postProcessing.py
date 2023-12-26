import SimpleITK as sitk
import numpy as np
import os
from skimage import measure

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

def postprocess(mask):
    # only keep the most biggest region
    labels = measure.label(mask, connectivity=3)
    max_num = 0
    max_pixel = -1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j
        print(np.sum(labels == j), np.sum(labels != 0))
        if np.sum(labels == j) > 0.1 * np.sum(labels != 0):
            labels[labels == j] = max_pixel
    if max_pixel==-1:
        print('No diff made')
        return mask
    else:
        labels[labels != max_pixel] = 0
        labels[labels == max_pixel] = 1
        return labels

def postprocess_remove_small_obj(mask,front_pixs,thres=0.008):
    # remove part if its smaller than front_pixs*thres.
    labels = measure.label(mask, connectivity=3)
    max_pixel = -1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) <= front_pixs*thres:
            labels[labels == j] = 0 # remove small dots
        else:
            labels[labels == j] =1
    if np.all(mask==labels):
        print('No diff made')
    return labels

def postprocess_remove_tumor_donot_connect_kidney(tumor_mask,kid_mask):
    # Now only have one biggest kidney. remove_tumor_donot_connect_kidney.
    tmp_mask = np.zeros_like(tumor_mask)
    tmp_mask[kid_mask == 1] = 1
    tmp_mask[tumor_mask == 1]=1
    labels = measure.label(tmp_mask, connectivity=1)
    max_num = 0
    max_pixel = -1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j
    if max_pixel==-1:
        print('No diff made')
    else:
        labels[labels != max_pixel] = 0
        labels[labels == max_pixel] = 1
    return labels

def postpro(pred_path,output_path):
    maybe_mkdir_p(output_path)

    for i in os.listdir(pred_path):
        if i[-3:]=='.gz':
            print(i,flush=True)
            case=i
            # read ct file, try to keep the maximum part, calculate the number of removed pixel.
            # save ct file for latter check.
            img = sitk.ReadImage(pred_path + case)
            arr = sitk.GetArrayFromImage(img)
            front = np.sum(arr == 1) + np.sum(arr == 2)
            spacing = img.GetSpacing()
            direct = img.GetDirection()
            empty_mask = np.zeros_like(arr)
            mask_kidney = np.zeros_like(arr)
            mask_kidney[arr == 1] = 1
            # mask_kidney = postprocess_remove_small_obj(mask_kidney,front)
            mask_kidney = postprocess(mask_kidney)
            mask_tumor = np.zeros_like(arr)
            mask_tumor[arr == 2] = 1
            mask_tumor_with_kids = postprocess_remove_tumor_donot_connect_kidney(mask_tumor, mask_kidney)
            # mask_tumor_tmp = np.zeros_like(arr)
            # mask_tumor_tmp[mask_tumor_with_kids == 1 and mask_tumor == 1] = 1
            # mask_tumor=mask_tumor_tmp
            mask_tumor[mask_tumor_with_kids == 1] += 1
            mask_tumor[mask_tumor == 1] = 0
            mask_tumor[mask_tumor == 2] = 1
            mask_tumor = postprocess_remove_small_obj(mask_tumor, front)
            empty_mask[mask_kidney == 1] = 1
            empty_mask[mask_tumor == 1] = 2
            im = sitk.GetImageFromArray(empty_mask)
            im.SetSpacing(spacing)
            im.SetDirection(direct)
            sitk.WriteImage(im,output_path + case)
            print('\tKidney before ', np.sum(arr == 1), ' , after ', np.sum(empty_mask == 1), ' , minus ',
                  np.sum(arr == 1) - np.sum(empty_mask == 1))
            kminus = np.sum(arr == 1) - np.sum(empty_mask == 1)
            print('\tTumor before ', np.sum(arr == 2), ' , after ', np.sum(empty_mask == 2), ' , minus ',
                  np.sum(arr == 2) - np.sum(empty_mask == 2))
            tminus = np.sum(arr == 2) - np.sum(empty_mask == 2)
            print('\tFront pixs ', front, ' kidney minus %.2f' % float(kminus / front * 100), '%',
                  ' tumor minus %.2f' % float(tminus / front * 100), '%')













