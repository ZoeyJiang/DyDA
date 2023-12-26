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


def json_G_kits(img_p,gt_p,out):

    cases = os.listdir(img_p)

    json_dict = {}
    json_dict['name'] = "Cecil"
    json_dict['description'] = "dyde"
    json_dict['tensorImageSize'] = "4D"
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
        if file.endswith('.nii.gz'):
        # case_00184_0000.nii.gz
        # case_1866861_0000.nii.gz
            if os.path.exists(gt_p + '/' + file[:-12] + '.nii.gz'):
                json_dict['training'].append(
                    {'image': "./ct/%s" % file[:-12]+'.nii.gz', "label": "./gt/%s" % file[:-12] + '.nii.gz'})
                count += 1
            else:
                json_dict['training'].append({'image': "./ct/%s" % file[:-12] + '.nii.gz', "label": "None"})
            # json_dict['training'].append({'image': "./img/%s" % file[:12]+'.nii.gz', "label": "./pseudo_gt/%s" % file[:12]+'.nii.gz'})

    # json_dict['numSource']=count
    json_dict['test'] = []

    filename = os.path.join(out, "dataset.json")
    with open(filename, 'w+') as file_obj:
        json.dump(json_dict, file_obj)

def json_G_acdc(img_p,gt_p,out):
    json_dict = {}
    json_dict['name'] = "ACDC_Myo_emidec"
    json_dict['description'] = "pseudo labels"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "pseudo labels data "
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MR",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "RV",
        "2": "LMB",
        "3": "LV"
    }
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "Kidney",
    #     "2": "Tumor"
    # }
    json_dict['numTraining'] = len(os.listdir(img_p))
    json_dict['numTest'] = 0
    json_dict['training'] = []
    count = 0
    for file in os.listdir(img_p):

        write_train_image = file[:-12] + '.nii.gz'
        if 'patient' in file:
            write_train_label = file[:-12] + '_gt.nii.gz'
        else:
            write_train_label = file[:-12]+'.nii.gz'
        if os.path.exists(gt_p + '/' + write_train_label):
            json_dict['training'].append(
                {'image': "./ct/%s" % write_train_image, "label": "./gt/%s" % write_train_label})
            count += 1
        else:
            json_dict['training'].append({'image': "./ct/%s" % write_train_image, "label": "None"})

    json_dict['numSource'] = count
    json_dict['test'] = []

    with open(os.path.join(out, "dataset.json"), 'w+') as file_obj:
        json.dump(json_dict, file_obj)

if __name__ == "__main__":
    img_p = '/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/do_dummy_2D_changed/lamda07/IntraDA/MinEnt/2/2_2/DyDA_dataset/dataset/ct'
    gt_p = '/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/do_dummy_2D_changed/lamda07/IntraDA/MinEnt/2/2_2/DyDA_dataset/dataset/gt'
    out = '/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/do_dummy_2D_changed/lamda07/IntraDA/MinEnt/2/2_2/DyDA_dataset/'


    json_G_acdc(img_p, gt_p,out)
