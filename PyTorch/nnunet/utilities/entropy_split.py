def jsph_kits_split(txt_path, IntraDA3d_lambda):
    # jzy: 手动排序entropy，分别根据待分割部位的entroy.txt进行排序，再划分比例
    # TODO：抽离成单独的函数
    # ------------Kidney part---------------------
    with open(txt_path + 'no_tumor_entropy.txt') as f:
        entropy_list = [case.strip().split(' ')[0] for case in f]

    # jzy: divide easy_hard split based on IntraDA3d_lambda
    easy_num = int(len(entropy_list) * IntraDA3d_lambda)
    easy_split = entropy_list[:easy_num]
    hard_split = entropy_list[easy_num:]
    print('Kidney easy dataset ', len(easy_split), ' cases : ', easy_split, flush=True)
    print('\nKidney hard dataset ', len(hard_split), ' cases : ', hard_split, flush=True)
    # -------------Tumor part---------------------
    with open(txt_path + 'with_tumor_entropy.txt') as f:
        entropy_list = [case.strip().split(' ')[0] for case in f]
    easy_num = int(len(entropy_list) * IntraDA3d_lambda)
    easy_split.extend(entropy_list[:easy_num])  # case_0041700
    hard_split.extend(entropy_list[easy_num:])
    print('\nTumor easy dataset ', easy_num, ' cases : ', entropy_list[:easy_num], flush=True)
    print('\nTumor hard dataset ', len(entropy_list) - easy_num, ' cases : ', entropy_list[easy_num:], flush=True)

    return easy_split, hard_split


def acdc_split(txt_path, IntraDA3d_lambda):
    # jzy: 手动排序entropy，分别根据待分割部位的entroy.txt进行排序，再划分比例
    # TODO：抽离成单独的函数
    # ------------Kidney part---------------------
    with open(txt_path + 'no_LV_entropy_list.txt') as f:
        entropy_list = [case.strip().split(' ')[0] for case in f]

    # jzy: divide easy_hard split based on IntraDA3d_lambda
    easy_num = int(len(entropy_list) * IntraDA3d_lambda)
    easy_split = entropy_list[:easy_num]
    hard_split = entropy_list[easy_num:]
    print('RV and LMB easy dataset ', len(easy_split), ' cases : ', easy_split, flush=True)
    print('\nRV and LMB hard dataset ', len(hard_split), ' cases : ', hard_split, flush=True)

    # -------------Tumor part---------------------
    with open(txt_path + 'with_LV_entropy_list.txt') as f:
        entropy_list = [case.strip().split(' ')[0] for case in f]
    easy_num = int(len(entropy_list) * IntraDA3d_lambda)
    easy_split.extend(entropy_list[:easy_num])  # case_0041700
    hard_split.extend(entropy_list[easy_num:])

    print('\nAll easy dataset ', easy_num, ' cases : ', entropy_list[:easy_num], flush=True)
    print('\nAll hard dataset ', len(entropy_list) - easy_num, ' cases : ', entropy_list[easy_num:], flush=True)

    return easy_split, hard_split
