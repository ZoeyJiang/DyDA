import os
import json
import shutil


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def json_G(img_p,gt_p,out):
    cases = os.listdir(img_p)

    json_dict = {}
    json_dict['name'] = "Cecil"
    json_dict['description'] = "pseudo labels"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "pseudo labels data for AdventCT"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor"
    }
    json_dict['numTraining'] = len(cases)
    json_dict['numTest'] = 0
    json_dict['training'] = []
    count = 0
    for file in os.listdir(img_p):
        # case_0011300_0000.nii.gz
        if os.path.exists(gt_p + '/' + file[:12] + '.nii.gz'):
            json_dict['training'].append(
                {'image': "./ct/%s" % file[:12] + '.nii.gz', "label": "./gt/%s" % file[:12] + '.nii.gz'})
            count += 1
        else:
            json_dict['training'].append({'image': "./ct/%s" % file[:12] + '.nii.gz', "label": "None"})
        # json_dict['training'].append({'image': "./img/%s" % file[:12]+'.nii.gz', "label": "./pseudo_gt/%s" % file[:12]+'.nii.gz'})
        print(file)

    # json_dict['numSource']=count
    json_dict['test'] = []

    filename = os.path.join(out, "dataset.json")
    with open(filename, 'w+') as file_obj:
        json.dump(json_dict, file_obj)
