#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys

sys.path.append("..")
sys.path.append("../..")
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier

plans_file_p = '/dataset/preprocessed/Task003_mixed_jph_kits/preprocessed_3D/nnUNetPlansv2.1_plans_3D.pkl'
dataset_p = '/dataset/preprocessed/Task003_mixed_jph_kits/preprocessed_3D/nnUNetData_plans_v2.1_stage0'
gt_p = '/dataset/preprocessed/Task003_mixed_jph_kits/preprocessed_3D/gt_segmentations/'
target_dataset_path_forvali = '/dataset/preprocessed/Task004_kits_vali/preprocessed_3D/nnUNetData_plans_v2.1_stage0/'
dynamicDA_targetdomain_ct_dataset = '/dataset/raw/raw_data/Task001_kits_train/imagesTr/'
dynamicDA_sourcedomain_dataset = '/dataset/raw/raw_data/Task002_jph/'

# --------3d param，效果比有2D时更好-----------------------------
source_train = False

only_ours = False  # train model using only source domain without UDA, should not be true

# only MinEnt
use_both_minent_and_advent_training = False  
using_adversarial_training = False or use_both_minent_and_advent_training

# only AdvEnt
# use_both_minent_and_advent_training = False  
# using_adversarial_training = True

# Both
# use_both_minent_and_advent_training = True
# using_adversarial_training = True

# No_adver
# use_both_minent_and_advent_training = True
# using_adversarial_training = False

# ------------don't change normally------------------------
# #--------Advent optimizer param---------
initial_lr = 0.00025
momentum = 0.9
weight_decay = 0.0005
# # --------nnUnet optimizer param(with default batchsize) ---------
# initial_lr = 0.01
# momentum = 0.99
# weight_decay = 0.00003
# #--------Adam optimizer param for Discriminator---
d_lr = 0.001
betas = (0.9, 0.99)


# -------IntraDA or DyDA ------
IntraDA_lambda = 0.3   # inital lamda
dynamicDA_target_lambda = 0.7   # final lambda

using_entropy_split = True  # using_entropy_split为F： 自動讀取 IntraDA_txt_dataset_path， 設置為空， T的時候是兩步訓練
IntraDA_txt_dataset_path = ''

DyDA = True  # T: DyDE or IntraDA, F: AdvEnt or MinEnt
With_origin_domain_training = True  #  Not used when using_entropy_split=false, don't change


# dynamic_split_epoch = dynamicDA_period， IntraDA_lambda = dynamicDA_target_lambda
dynamicDA_period = 10  
dynamic_split_epoch = 2  


output_folder = '/dataset/trained_models/Task003_mixed_jph_kits/0-params/tumor_kidney/1_1/2_split_10_period'


# when continue_training = T：use model from the latest_model_path
continue_training = True
using_dynamicDA_after_train = True

latest_model_path = '/dataset/trained_models/Task003_mixed_jph_kits/dyde/model_latest.model'

# #--------Dynamic DA param---------
dynamicDA_initial_lr = 0.0035  # initial_lr / (inter_epoch/dynamicDA_period) + x
# dynamicDA_lambda = IntraDA_lambda + (dynamicDA_target_lambda-IntraDA_lambda)/dynamicDA_cycle * (current_cycle+1)
dynamicDA_temporary_dataset_dir = output_folder + '/DyDA_dataset/'   # don't change temp path
batchsize_source = 5  # 2 for only_ours and new crop algorithm,  5 for old crop dataset UDA
batchsize_target = 5  # 2 for only_ours

max_num_epochs = 10
train_epochs = 250*max_num_epochs

weight_unsupervised_loss = 0.01 
weight_discriminator_loss = 0.0007  
source_training_augmentation = True  
using_ranger_optimizer = False

#   # ------------------Weighted param-----------------
weighted_entropy_rank = True



def main():
    network = '3d_fullres'
    network_trainer = 'nnUNetTrainerV2'
    plans_identifier = default_plans_identifier

    plans_file, output_folder_name, \
    dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(
        dataset_p=dataset_p,
        plans_file=plans_file_p,
        output_folder=output_folder,
        network=network,
        network_trainer=network_trainer,
        plans_identifier=plans_identifier
    )

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")


    trainer = trainer_class(target_dataset_path_forvali, train_epochs, source_training_augmentation, batchsize_source,
                            batchsize_target, gt_p, plans_file, 'all', output_folder=output_folder_name,
                            dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=True,
                            deterministic=False,
                            fp16=True)
    print('debug trainer_class before initializing in run_training:', trainer_class)

    trainer.initialize(False)

    # 训练到一半停止的模型可以设置was_initialized=False
    # trainer.initialize(False)

    print('debug trainer_class after initializing in run_training:', trainer_class, type(trainer_class))

    if continue_training:
        trainer.load_latest_checkpoint(latest_model_path)

    trainer.run_training()

    trainer.network.eval()

    # # predict validation
    # trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)


if __name__ == "__main__":
    main()
