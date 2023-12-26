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
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

plans_file_p = '/dataset/preprocessed/Task011_new_mixed_ACDC_myo_emidec/preprocessed/nnUNetPlansv2.1_plans_3D_do_dummy_2D_data_augChanged.pkl'
dataset_p = '/dataset/preprocessed/Task011_new_mixed_ACDC_myo_emidec/preprocessed/nnUNetData_plans_v2.1_stage0'
gt_p = '/dataset/preprocessed/Task011_new_mixed_ACDC_myo_emidec/preprocessed/gt_segmentations'

target_dataset_path_forvali = '/dataset/preprocessed/Task013_new_mixed_vali/preprocessed/nnUNetData_plans_v2.1_stage0'
dynamicDA_targetdomain_ct_dataset = '/dataset/raw/raw_data/Task012_myo_emidec/imagesTr/'
dynamicDA_sourcedomain_dataset = '/dataset/raw/raw_data/Task010_ACDC/'


only_ours = False  # train model using only source domain without UDA, should not be true

# MinEnt
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

# ----------------don't change ----------------
# #--------Advent optimizer param---------
# initial_lr = 0.00025
# momentum = 0.9
# weight_decay = 0.0005
# # --------nnUnet optimizer param(with default batchsize) keep---------
initial_lr = 0.01
momentum = 0.99
weight_decay = 0.00003
# #--------Adam optimizer param for Discriminator ------
d_lr = 0.001
betas = (0.9, 0.99)


IntraDA_lambda = 0.3   # inital lambda
dynamicDA_target_lambda = 0.7   # final lambda

using_entropy_split = True  
IntraDA_txt_dataset_path = ''

using_dyncmicDA_after_train = True  # T: DyDE or IntraDA, F: AdvEnt or MinEnt
With_origin_domain_training = True  


# dynamic_split_epoch = dynamicDA_period， IntraDA_lambda = dynamicDA_target_lambda
dynamicDA_period = 50  
dynamic_split_epoch = 20 


output_folder = '/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/do_dummy_2D_changed/lamda07/DyDA_Both/'

# when continue_training = T：use model from the latest_model_path
continue_training = True
latest_model_path = '/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/do_dummy_2D_changed/lamda07/Both_True/continueTr/model_final_checkpoint.model'


# #--------Dynamic DA param---------
dynamicDA_initial_lr = 0.0035  
dynamicDA_temporary_dataset_dir = output_folder + '/DyDA_dataset/'   
batchsize_source = 5  
batchsize_target = 5  
train_epochs = 250000 
weight_unsupervised_loss = 0.01  
weight_discriminator_loss = 0.0007  
source_training_augmentation = True  
using_ranger_optimizer = True  


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("network")
    # parser.add_argument("network_trainer")
    # parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
    #                     default=default_plans_identifier, required=False)
    # args = parser.parse_args()
    # network = args.network
    # network_trainer = args.network_trainer
    # plans_identifier = args.p

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
    # trainer_class = get_default_configuration(dataset_p, plans_file_p, output_folder, network, network_trainer)
    # dataset_p, plans_file, output_folder, network_trainer, network = '3d_fullres'

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    assert issubclass(trainer_class,
                      nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

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
