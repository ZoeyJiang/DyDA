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


from collections import OrderedDict
from typing import Tuple
import torch.nn.functional as F

import numpy as np
import torch
from nnunet.run.run_training import initial_lr, momentum, weight_decay,dynamicDA_targetdomain_ct_dataset,\
    using_adversarial_training,d_lr,using_entropy_split,IntraDA_txt_dataset_path,using_ranger_optimizer,weight_unsupervised_loss,\
    weight_discriminator_loss,use_both_minent_and_advent_training,DyDA,dynamicDA_period,dynamic_split_epoch,\
    dynamicDA_initial_lr,dynamicDA_target_lambda,dynamicDA_temporary_dataset_dir,dynamicDA_sourcedomain_dataset,only_ours,IntraDA_lambda
from nnunet.training.loss_functions.deep_supervision import *
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation, get_no_augmentation, \
    get_default_augmentation, get_moreDA_augmentation_tr_only,get_no_augmentation_tr_only
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.discriminator import *
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import torch.backends.cudnn as cudnn
from ..optimizer.ranger import Ranger
from nnunet.run.run_training import With_origin_domain_training, weighted_entropy_rank

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    if prob.dim()==4:
        n, c, h, w = prob.size()
    if prob.dim()==5:
        n, c, h, w, t = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, vali_datasetfolder, epoch, source_training_augmentation, bs_source, bs_target, gt_p, plans_file,
                 fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(gt_p, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        # self.max_num_epochs = epoch
        self.max_num_epochs = epoch//self.num_batches_per_epoch
        self.vali_datasetfolder = vali_datasetfolder
        self.initial_lr = initial_lr
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.bs_source = bs_source
        self.bs_target = bs_target
        self.source_training_augmentation = source_training_augmentation
        self.DyDA = DyDA
        self.model_dir_path = output_folder
        if DyDA:
            self.IntraDA_lambda = IntraDA_lambda
            self.dynamicDA_period = dynamicDA_period
            self.dynamic_split_epoch = dynamic_split_epoch
            self.dynamicDA_target_lambda = dynamicDA_target_lambda
            self.dynamicDA_temporary_dataset_dir = dynamicDA_temporary_dataset_dir
            self.dynamicDA_sourcedomain_dataset = dynamicDA_sourcedomain_dataset
            self.dynamicDA_targetdomain_ct_dataset = dynamicDA_targetdomain_ct_dataset
            self.dynamicDA_initial_lr = dynamicDA_initial_lr

        if using_adversarial_training:
            self.using_adversarial_training=True
        else:
            self.using_adversarial_training=False

        self.pin_memory = True
        self.weighted_entropy_rank = weighted_entropy_rank


    def reset_dataset_path(self, new_dataset_preprocessed, new_gt_p, Dyda_txt_path, DyDA_lambda):
        # self.reset_dataset_path(
        #     self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed/nnUNetData_plans_v2.1_stage0',
        #     self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed/gt_segmentations',
        #     self.dynamicDA_temporary_dataset_dir + dynamic_entropy_name,
        #     self.IntraDA_lambda + (self.dynamicDA_target_lambda - self.IntraDA_lambda) / self.dynamic_split_epoch * (x + 1)
        #     )
        self.print_to_log_file('=========================\nCurrent lambda ', DyDA_lambda, "\n=========================")
        self.dataset_directory = new_dataset_preprocessed
        self.gt_niftis_folder = new_gt_p #not quite necessary but whatever
        self.folder_with_preprocessed_data = self.dataset_directory


        # using_entropy_split = True & source_training_augmentation = True
        # get_basic_generators in nnUNetTrainer.py
        self.dl_tr, self.dl_target, self.dl_target_val = self.get_basic_generators(self.bs_source,
                                                                                   self.bs_target,
                                                                                   using_entropy_split=True,
                                                                                   txt_path=Dyda_txt_path,
                                                                                   IntraDA3d_lambda=DyDA_lambda,
                                                                                   With_origin_domain_training=With_origin_domain_training)
        unpack_dataset(self.folder_with_preprocessed_data, overwrite=True)
        self.target_gen = get_moreDA_augmentation_tr_only(self.dl_target,
                                                          self.data_aug_params[
                                                              'patch_size_for_spatialtransform'],
                                                          self.data_aug_params,
                                                          deep_supervision_scales=self.deep_supervision_scales,
                                                          pin_memory=self.pin_memory)
        self.tr_gen, self.target_val_gen = get_moreDA_augmentation(
            self.dl_tr, self.dl_target_val,
            self.data_aug_params[
                'patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory
        )


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision"""
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)
            self.load_plans_file()
            self.process_plans(self.plans)
            # self.do_dummy_2D_aug = False # test


            self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
            if self.threeD: # 3D 数据数据增强 TODO: 可能删除或修改rotation_xyz
                self.data_aug_params = default_3D_augmentation_params
                self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
                self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
                self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
                if self.do_dummy_2D_aug:
                    self.data_aug_params["dummy_2D"] = True
                    self.print_to_log_file("Using dummy2d data augmentation")
                    self.data_aug_params["elastic_deform_alpha"] = \
                        default_2D_augmentation_params["elastic_deform_alpha"]
                    self.data_aug_params["elastic_deform_sigma"] = \
                        default_2D_augmentation_params["elastic_deform_sigma"]
                    self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
            else:
                self.do_dummy_2D_aug = False
                if max(self.patch_size) / min(self.patch_size) > 1.5:
                    default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
                self.data_aug_params = default_2D_augmentation_params
            self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size
            # nnUNet 特有数据增强方式，不需要修改
            self.data_aug_params["scale_range"] = (0.7, 1.4)
            self.data_aug_params["do_elastic"] = False
            self.data_aug_params['selected_seg_channels'] = [0]
            self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform
            self.data_aug_params["num_cached_per_thread"] = 2

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            # TODO: 记录到文档中
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            self.entropy_loss = MultipleOutputEntropyLoss(self.ds_loss_weights)  # key point of unsupervised learning
            ################# END ###################


            # TODO: 根据参数整理成utils.funs
            if training:
                self.folder_with_preprocessed_data = self.dataset_directory
                # def get_basic_generators(self, batch_size_source=1, batch_size_target=1, using_entropy_split=False, txt_path='',
                #                              IntraDA3d_lambda=None,With_origin_domain_training=None,only_ours=False):
                if only_ours:
                    self.dl_tr, self.dl_target, self.dl_target_val = self.get_basic_generators(self.bs_source,
                                                                                               self.bs_target,
                                                                                               using_entropy_split,
                                                                                               IntraDA_txt_dataset_path,only_ours=True)
                else:
                    if using_entropy_split:
                        from nnunet.run.run_training import IntraDA_lambda,With_origin_domain_training
                        self.IntraDA_lambda = IntraDA_lambda
                        self.dl_tr, self.dl_target, self.dl_target_val = self.get_basic_generators(self.bs_source, self.bs_target,using_entropy_split,IntraDA_txt_dataset_path,IntraDA_lambda,With_origin_domain_training)
                    else:
                        self.dl_tr, self.dl_target, self.dl_target_val = self.get_basic_generators(self.bs_source, self.bs_target,using_entropy_split,IntraDA_txt_dataset_path)
                unpack_dataset(self.folder_with_preprocessed_data)

                if only_ours:
                    if (self.source_training_augmentation):
                        self.target_gen = None
                        self.tr_gen, self.target_val_gen = get_moreDA_augmentation(
                            self.dl_tr, self.dl_target_val,
                            self.data_aug_params[
                                'patch_size_for_spatialtransform'],
                            self.data_aug_params,
                            deep_supervision_scales=self.deep_supervision_scales,
                            pin_memory=self.pin_memory
                        )
                else:
                    if (self.source_training_augmentation):
                        # self.target_gen=get_no_augmentation(self.dl_target,
                        self.target_gen = get_moreDA_augmentation_tr_only(self.dl_target,
                                                                          self.data_aug_params[
                                                                              'patch_size_for_spatialtransform'],
                                                                          self.data_aug_params,
                                                                          deep_supervision_scales=self.deep_supervision_scales,
                                                                          pin_memory=self.pin_memory)
                        self.tr_gen, self.target_val_gen = get_moreDA_augmentation(
                            self.dl_tr, self.dl_target_val,
                            self.data_aug_params[
                                'patch_size_for_spatialtransform'],
                            self.data_aug_params,
                            deep_supervision_scales=self.deep_supervision_scales,
                            pin_memory=self.pin_memory
                        )
                    else:
                        self.target_gen = get_no_augmentation_tr_only(self.dl_target,
                                                                          self.data_aug_params[
                                                                              'patch_size_for_spatialtransform'],
                                                                          self.data_aug_params,
                                                                          deep_supervision_scales=self.deep_supervision_scales,
                                                                          pin_memory=self.pin_memory)
                        self.tr_gen, self.target_val_gen = get_no_augmentation(
                            self.dl_tr, self.dl_target_val,
                            self.data_aug_params[
                                'patch_size_for_spatialtransform'],
                            self.data_aug_params,
                            deep_supervision_scales=self.deep_supervision_scales,
                            pin_memory=self.pin_memory
                        )
            else:
                pass
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        # ADVENT_UDA combine feature_level and output_level
        if using_adversarial_training: # UDA_ADVENT
            #jzy: feature-level
            self.d_aux = get_fc_discriminator(num_classes=self.num_classes,threeD=self.threeD)
            self.d_aux.train()
            self.d_aux.cuda()

            # output-level
            self.d_main = get_fc_discriminator(num_classes=self.num_classes,threeD=self.threeD)
            self.d_main.train()
            self.d_main.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if using_ranger_optimizer:
            self.optimizer = Ranger(self.network.parameters(), lr=initial_lr)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4,eta_min=4e-08)
            self.optimizer_d_main = Ranger(self.network.parameters(), lr=d_lr)
            self.d_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d_main, T_max=5, eta_min=4e-08)

        # segnet's optimizer
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), initial_lr, weight_decay=weight_decay,
                                             momentum=momentum, nesterov=True)

            # ADVENT_UDA combine output_leve and feature leve
            if using_adversarial_training:
                # feature level
                # self.optimizer_d_aux = torch.optim.Adam(self.d_aux.parameters(), lr=d_lr, betas=(0.9,0.99))
                # output level
                self.optimizer_d_main = torch.optim.Adam(self.d_main.parameters(), lr=d_lr,
                                                        betas=(0.9, 0.99))
            else:
                self.optimizer_d_main = Ranger(self.network.parameters(), lr=d_lr)


    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                               overwrite, validation_folder_name, debug, all_in_gpu, segmentation_export_kwargs)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = True,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes,
                                                                       use_sliding_window, step_size, use_gaussian,
                                                                       pad_border_mode, pad_kwargs, all_in_gpu, verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, target_data_generator, do_backprop=True, run_online_evaluation=False, adv_training=False):
        # data_dict只需要其中 data、target变量而已
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # new added do_backprop
        if do_backprop:
            if target_data_generator is not None:
                targetdomain_dict = next(target_data_generator)
            else:
                targetdomain_dict = data_dict.copy()

            target_img = targetdomain_dict['data']
            target_img = maybe_to_torch(target_img)

            # original judgement
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)
                target_img = to_cuda(target_img)  # new added target_img

            self.optimizer.zero_grad()

            if only_ours:
                with autocast():
                    output = self.network(data)
                    l1 = self.loss(output, target)

                if do_backprop:
                    self.amp_grad_scaler.scale(l1).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer) # new added
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12) # new added
                    self.amp_grad_scaler.step(self.optimizer) # new added
                    self.amp_grad_scaler.update()
                return l1.detach().cpu().numpy(), 0.1   # new added
            else:
                if use_both_minent_and_advent_training:
                    self.optimizer_d_main.zero_grad()
                    # UDA Training
                    # only train segnet. Don't accumulate grads in disciminators
                    for param in self.d_main.parameters():
                        param.requires_grad = False
                    # training seg net with gt first
                    with autocast():
                        output = self.network(data)
                        l1 = self.loss(output, target)
                    self.amp_grad_scaler.scale(l1).backward()

                    # MinEnt training using entropy_loss
                    with autocast():
                        target_pred = self.network(target_img)
                        entropy_l = self.entropy_loss(target_pred) * weight_unsupervised_loss
                    self.amp_grad_scaler.scale(entropy_l).backward()

                    # adversarial training to fool the D
                    with autocast():
                        target_pred = self.network(target_img)
                        d_out_main = self.d_main(prob_2_entropy(F.softmax(target_pred[0],
                                                                          dim=1)))  # we can only use one network to discriminate one entropy map, forget deep-supervision
                        loss_adv_trg_main = bce_loss(d_out_main, self.source_label)
                        l2 = weight_discriminator_loss * loss_adv_trg_main
                    self.amp_grad_scaler.scale(l2).backward()

                    # ---------------Train discriminator networks--------------
                    # enable training mode on discriminator networks
                    for param in self.d_main.parameters():
                        param.requires_grad = True
                    # train with source
                    pred_src_main = output[0].detach()
                    with autocast():
                        d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
                        loss_d_main_s = bce_loss(d_out_main, self.source_label)
                        loss_d_main_s = loss_d_main_s / 2
                    self.amp_grad_scaler.scale(loss_d_main_s).backward()
                    # train with target
                    pred_trg_main = target_pred[0].detach()
                    with autocast():
                        d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
                        loss_d_main_t = bce_loss(d_out_main, self.target_label)
                        loss_d_main_t = loss_d_main_t / 2
                    self.amp_grad_scaler.scale(loss_d_main_t).backward()

                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
                    self.amp_grad_scaler.unscale_(self.optimizer_d_main)
                    torch.nn.utils.clip_grad_norm_(self.d_main.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer_d_main)
                    self.amp_grad_scaler.update()

                    return l1.detach().cpu().numpy(), loss_adv_trg_main.detach().cpu().numpy(), loss_d_main_s.detach().cpu().numpy(), loss_d_main_t.detach().cpu().numpy()
                else:
                    if adv_training:
                        # reset optimizers
                        # self.optimizer_d_aux.zero_grad()
                        self.optimizer_d_main.zero_grad()

                        # only train segnet
                        # for param in self.d_aux.parameters():
                        #     param.requires_grad = False
                        for param in self.d_main.parameters():
                            param.requires_grad = False
                        with autocast():
                            output = self.network(data)
                            l1 = self.loss(output, target)
                        self.amp_grad_scaler.scale(l1).backward()

                        # adversairal training to fool the discriminator(G-UNet structure)
                        with autocast():
                            target_pred = self.network(target_img)
                            d_out_main = self.d_main(prob_2_entropy(F.softmax(target_pred[0],dim=1)))  # we can only use one network to discriminate one entropy map, forget deep-supervision
                            loss_adv_trg_main = bce_loss(d_out_main, self.source_label)
                            l2 = weight_discriminator_loss * loss_adv_trg_main
                        self.amp_grad_scaler.scale(l2).backward()

                        # -----train discriminator network------
                        # for param in self.d_aux.parameters():
                        #     param.requires_grad = True
                        for param in self.d_main.parameters():
                            param.requires_grad = True
                        pred_src_main = output[0].detach()
                        # train with source
                        with autocast():
                            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main,dim=1)))
                            loss_d_main_s = bce_loss(d_out_main, self.source_label)
                            loss_d_main_s = loss_d_main_s / 2
                        self.amp_grad_scaler.scale(loss_d_main_s).backward()
                        # train with target
                        pred_trg_main = target_pred[0].detach()
                        with autocast():
                            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
                            loss_d_main_t = bce_loss(d_out_main, self.target_label)
                            loss_d_main_t = loss_d_main_t / 2
                        self.amp_grad_scaler.scale(loss_d_main_t).backward()

                        self.amp_grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                        self.amp_grad_scaler.step(self.optimizer)
                        self.amp_grad_scaler.update()
                        self.amp_grad_scaler.unscale_(self.optimizer_d_main)
                        torch.nn.utils.clip_grad_norm_(self.d_main.parameters(), 12)
                        self.amp_grad_scaler.step(self.optimizer_d_main)
                        self.amp_grad_scaler.update()

                        return l1.detach().cpu().numpy(), loss_adv_trg_main.detach().cpu().numpy(),loss_d_main_s.detach().cpu().numpy(),loss_d_main_t.detach().cpu().numpy()
                    else:
                        # UDA Training
                        # only train segnet. Donnot accumulate grads in discimators
                        with autocast(): #用于加速半精度训练
                            output = self.network(data)
                            l1 = self.loss(output, target)  # wrong position, self.loss
                        if do_backprop:
                            self.amp_grad_scaler.scale(l1).backward()

                        # adversarial training with minent
                        with autocast():
                            target_pred = self.network(target_img)  # already have softmax layer in the network
                            del target_img
                            entropy_l = self.entropy_loss(target_pred) * weight_unsupervised_loss
                        if do_backprop:
                            self.amp_grad_scaler.scale(entropy_l).backward()

                            self.amp_grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                            self.amp_grad_scaler.step(self.optimizer)
                            self.amp_grad_scaler.update()

                        return l1.detach().cpu().numpy(), entropy_l.detach().cpu().numpy() / weight_unsupervised_loss
        else:
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()

            if self.fp16:
                with autocast():
                    output = self.network(data)
                    del data
                    l = self.loss(output, target)

            else:
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if run_online_evaluation:
                self.run_online_evaluation(output, target)

            del target

            return l.detach().cpu().numpy()

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        if using_ranger_optimizer: #ranger optimizer is bind with lr_scheduler
            # self.lr_scheduler.step(ep)   # jzy：no diffs without the step to self.lr_schedduler
            if using_adversarial_training:
                self.d_lr_scheduler.step(ep)
        else:
            # 定义优化器参数
            self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            if using_adversarial_training:
                self.optimizer_d_main.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, d_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def maybe_update_lr_DyDA(self, epoch=None,on_cycle_start=True):
        if on_cycle_start:
            ep = 1
        else:
            ep = ((self.epoch)-self.max_num_epochs) % self.dynamicDA_period
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.dynamicDA_period+1, self.dynamicDA_initial_lr, 0.9)
        if using_adversarial_training:
            self.optimizer_d_main.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, d_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        # if self.epoch == 100:
        #     if self.all_val_eval_metrics[-1] == 0:
        #         self.optimizer.param_groups[0]["momentum"] = 0.95
        #         self.network.apply(InitWeights_He(1e-2))
        #         self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
        #                                "high momentum. High momentum (0.99) is good for datasets where it works, but "
        #                                "sometimes causes issues such as this one. Momentum has now been reduced to "
        #                                "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        print('begin run_training in nnUNetTrainerV2:')
        self.network.train()
        cudnn.benchmark = True
        cudnn.enabled = True
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret


