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


from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
# from nnunet.utilities.visualiz import plot_dynamic_boundary
from nnunet.run.run_training import using_adversarial_training
import shutil

def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path) and 'dy_' not in f:
            os.remove(file_path)
        elif os.path.isdir(file_path) and 'dy_' not in file_path:
            shutil.rmtree(file_path)


def copy_search_file(srcDir, desDir):
    ls = os.listdir(srcDir)
    for line in ls:
        filePath = os.path.join(srcDir, line)
        if os.path.isfile(filePath):
            shutil.copy(filePath, desDir)

class NetworkTrainer(object):
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        self.fp16 = fp16
        self.amp_grad_scaler = None
        cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.fold = None
        self.loss = None
        self.dataset_directory = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        self.dataset_tr = self.dataset_val = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 10
        self.num_batches_per_epoch = 250
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_tr_target_entropy_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = False
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        :param training:
        :return:
        """

    @abstractmethod
    def load_dataset(self):
        pass

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        tr_keys = val_keys = list(self.dataset.keys())
        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        # self.dataset_val = OrderedDict()
        # for i in val_keys:
        #     self.dataset_val[i] = self.dataset[i]

    def plot_progress(self):
        """
        Should probably by improved
        :return:

                self.all_d_segloss.append(np.mean(train_d_loss))
                self.train_d_Ours.append(np.mean(train_d_Ours))
                self.train_d_Kits.append(np.mean(train_d_Kits))
        """
        from nnunet.run.run_training import using_adversarial_training, d_lr
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")
            if using_adversarial_training:
                if len(x_values) != len(self.all_d_segloss):
                    self.all_d_segloss=[0]*len(x_values)
                if len(x_values) != len(self.train_d_Ours):
                    self.train_d_Ours = [0] * len(x_values)
                if len(x_values) != len(self.train_d_Kits):
                    self.train_d_Kits = [0] * len(x_values)
                ax.plot(x_values, self.all_d_segloss, color='r', ls='-', label="all_d_segloss")
                ax.plot(x_values, self.train_d_Ours, color='purple', ls='-', label="train_d_Ours")
                ax.plot(x_values, self.train_d_Kits, color='yellow', ls='-', label="train_d_Kits")
            else:
                if len(x_values) != len(self.all_tr_target_entropy_losses):
                    self.all_tr_target_entropy_losses=[self.all_tr_target_entropy_losses[0]]*len(x_values)
                ax.plot(x_values, self.all_tr_target_entropy_losses, color='r', ls='-', label="target_entropy")

            ax.plot(x_values, self.all_val_losses, color='black', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        if self.using_adversarial_training:
            #     save discriminator network
            d_state_dict = self.d_main.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        if self.using_adversarial_training:
            for key in d_state_dict.keys():
                d_state_dict[key] = d_state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
            if self.using_adversarial_training:
                d_optimizer_state_dict = self.optimizer_d_main.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        if self.using_adversarial_training:
            print('------Saving adv net-------',flush=True)
            save_this = {
                'epoch': self.epoch + 1,
                'state_dict': state_dict,
                'd_state_dict': d_state_dict,
                'd_optimizer_state_dict': d_optimizer_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_sched_state_dct,
                'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                               self.all_val_eval_metrics)}
        else:
            save_this = {
                'epoch': self.epoch + 1,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_sched_state_dct,
                'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                               self.all_val_eval_metrics)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, latest_model_path, train=True):
        return self.load_checkpoint(latest_model_path, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        pass

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        if self.using_adversarial_training:
            new_d_state_dict = OrderedDict()
            curr_d_state_dict_keys = list(self.d_main.state_dict().keys())
            for k, value in checkpoint['d_state_dict'].items():
            # for k, value in checkpoint['state_dict'].items():
                key = k
                if key not in curr_d_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                new_d_state_dict[key] = value

        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        if self.using_adversarial_training:
            self.d_main.load_state_dict(new_d_state_dict)
        # self.epoch = 0  # start over
        self.epoch = checkpoint['epoch']

        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if self.using_adversarial_training:
                d_optimizer_state_dict = checkpoint['d_optimizer_state_dict']
                if d_optimizer_state_dict is not None:
                    print('Load adv optimizer state', flush=True)
                    self.optimizer_d_main.load_state_dict(d_optimizer_state_dict)
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None and torch.cuda.is_available():
            self.amp_grad_scaler = GradScaler()

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def DyDA_UDA(self):
        from nnunet.inference.predict import predict_from_folder, check_input_folder_and_return_caseIDs, \
            preprocess_multithreaded
        from multiprocessing import Pool
        from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
        from nnunet.mypostprocess.postProcessing import postpro_kits, postpro_acdc
        from nnunet.utilities.entropy_ranking_3d import rank_acdc_myo_emidec, rank_jsph_kits
        from nnunet.utilities.json_gen import json_G_acdc, json_G_kits
        from nnunet.experiment_planning.nnUNet_plan_and_preprocess import preprocess_3d
        from nnunet.run.run_training import With_origin_domain_training
        from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, \
            save_segmentation_nifti

        maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir)

        for x in range(self.dynamic_split_epoch):
            # update gt and preprocess files in for-loop
            if x != 0:
                del_file(self.dynamicDA_temporary_dataset_dir[:-1])
            dynamic_entropy_name = 'dy_entropy_rank_' + str(x)
            dynamic_predict_name = 'dy_predict_result_' + str(x)
            dynamic_postprocess = 'dy_postprocess_' + str(x)

            expected_num_modalities = load_pickle(join(self.model_dir_path, "plans.pkl"))['num_modalities']

            case_ids = check_input_folder_and_return_caseIDs(self.dynamicDA_targetdomain_ct_dataset,
                                                             expected_num_modalities)     # target_case_ids: case_0000
            output_files = [join(self.dynamicDA_temporary_dataset_dir + dynamic_predict_name, i + ".nii.gz") for i in
                            case_ids]          # target_output_files
            all_files = subfiles(self.dynamicDA_targetdomain_ct_dataset, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [
                [join(self.dynamicDA_targetdomain_ct_dataset, i) for i in all_files if i[:len(j)].startswith(j) and
                 len(i) == (len(j) + 12)] for j in case_ids]
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
            cleaned_output_files = []
            for o in output_files:
                dr, f = os.path.split(o)
                if len(dr) > 0:
                    maybe_mkdir_p(dr)
                if not f.endswith(".nii.gz"):
                    f, _ = os.path.splitext(f)
                    f = f + ".nii.gz"
                cleaned_output_files.append(join(dr, f))


            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + dynamic_entropy_name)
            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + dynamic_predict_name)
            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + 'dataset/ct')
            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + 'dataset/gt')
            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/cropped')
            maybe_mkdir_p(self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed')

            print('dynamicDA_targetdomain_ct_dataset in network_trainer:', self.dynamicDA_targetdomain_ct_dataset)

            copy_search_file(self.dynamicDA_targetdomain_ct_dataset,
                             self.dynamicDA_temporary_dataset_dir + 'dataset/ct')

            # 1. Set new initial lr. Predict and Save result.
            # 2. Process result. Include postprocess, preprocess , entropy rank
            # 3. Set new dataset dir path. Get new batch generators

            self.maybe_update_lr_DyDA(self.epoch, True)
            # predict and save result
            self.print_to_log_file('--------------------------------------\nDCSS cycle ', (x + 1),
                                   '\nStart inference......')
            self.network.eval()
            pool = Pool(2)
            results = []
            torch.cuda.empty_cache()
            preprocessing = preprocess_multithreaded(self, list_of_lists, cleaned_output_files, num_processes=6)


            all_output_files = []
            for preprocessed in preprocessing:
                output_filename, (d, dct) = preprocessed
                all_output_files.append(all_output_files)
                if isinstance(d, str):
                    data = np.load(d)
                    os.remove(d)
                    d = data
                # output_folder/DyDA_dataset/predict_result/myops_training_116_T2.nii.gz
                softmax = []
                softmax.append(self.predict_preprocessed_data_return_seg_and_softmax(d, True, self.data_aug_params[
                    'mirror_axes'], True, step_size=0.5, use_gaussian=True, all_in_gpu=None, mixed_precision=True)[
                                   1][None])
                softmax = np.vstack(softmax)
                softmax_mean = np.mean(softmax, 0)
                transpose_forward = self.plans.get('transpose_forward')
                if transpose_forward is not None:
                    transpose_backward = self.plans.get('transpose_backward')
                    softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])
                # Save npz_file
                npz_file = output_filename[:-7] + ".npz"

                if hasattr(self, 'regions_class_order'):
                    region_class_order = self.regions_class_order
                else:
                    region_class_order = None
                bytes_per_voxel = 4
                if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
                    np.save(output_filename[:-7] + ".npy", softmax_mean)
                    softmax_mean = output_filename[:-7] + ".npy"
                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                  ((softmax_mean, output_filename, dct, interpolation_order,
                                                    region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z),)
                                                  ))
            self.print_to_log_file('inference done. Now waiting for the segmentation export to finish...')
            # results contains all target imageTr and

            for i in results:
                self.print_to_log_file(i)
            _ = [i.get() for i in results]
            results = []

            self.print_to_log_file('before postprocessing.....')
            pool.close()
            pool.join()

            # process result
            # postprocess
            postpro_kits(self.dynamicDA_temporary_dataset_dir + dynamic_predict_name + '/',
                         self.dynamicDA_temporary_dataset_dir + dynamic_postprocess + '/')
            rank_jsph_kits(weighted_entropy_rank=self.weighted_entropy_rank, npz_path=self.dynamicDA_temporary_dataset_dir + dynamic_predict_name,
                           postprocessd_ct_path=self.dynamicDA_temporary_dataset_dir + dynamic_predict_name,
                           outtxt_path=self.dynamicDA_temporary_dataset_dir + dynamic_entropy_name)

            # mix source_raw_label and target_raw_postprocessedPred
            copy_search_file(self.dynamicDA_sourcedomain_dataset + 'imagesTr',
                                 self.dynamicDA_temporary_dataset_dir + 'dataset/ct')
            copy_search_file(self.dynamicDA_sourcedomain_dataset + 'labelsTr',
                             self.dynamicDA_temporary_dataset_dir + 'dataset/gt')
            copy_search_file(self.dynamicDA_temporary_dataset_dir + dynamic_postprocess,
                             self.dynamicDA_temporary_dataset_dir + 'dataset/gt')
            # preprocess
            json_G_kits(self.dynamicDA_temporary_dataset_dir + 'dataset/ct',
                   self.dynamicDA_temporary_dataset_dir + 'dataset/gt',
                   self.dynamicDA_temporary_dataset_dir + 'dataset')

            preprocess_3d(True, self.dynamicDA_temporary_dataset_dir + 'dataset/dataset.json',
                          self.dynamicDA_temporary_dataset_dir + 'dataset/ct',
                          self.dynamicDA_temporary_dataset_dir + 'dataset/gt',
                          self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/cropped',
                          self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed',
                          plan_path=self.init_args[0])

            # Set new dataset dir path, get new generators
            # DyDA  reset in nnUNetTrainerV2.py
            self.reset_dataset_path(
                self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed/nnUNetData_plans_v2.1_stage0',
                self.dynamicDA_temporary_dataset_dir + 'dataset/preprocess/preprocessed/gt_segmentations',
                self.dynamicDA_temporary_dataset_dir + dynamic_entropy_name + '/',
                self.IntraDA_lambda + (self.dynamicDA_target_lambda - self.IntraDA_lambda) / self.dynamic_split_epoch * (x + 1)
            )

            # Start DCSS training
            for ce in range(self.dynamicDA_period):
                self.print_to_log_file("\nepoch: ", self.epoch)
                epoch_start_time = time()
                train_losses_epoch = []
                train_target_entropy_loss = []
                train_d_loss = []
                train_d_Ours = []
                train_d_Kits = []
                # train one epoch
                self.network.train()
                # if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("iter {}/{}".format(ce + 1, self.dynamicDA_period))
                        if self.using_adversarial_training:
                            l, dlossonseg, dlossonOurs, dlossonKits = self.run_iteration(self.tr_gen,
                                                                                         self.target_gen, True,
                                                                                         False, True)
                            tbar.set_postfix(loss=l)
                            train_losses_epoch.append(l)
                            train_d_loss.append(dlossonseg)
                            train_d_Ours.append(dlossonOurs)
                            train_d_Kits.append(dlossonKits)
                        else:
                            l, entropy_l = self.run_iteration(self.tr_gen, self.target_gen, True, False)
                            tbar.set_postfix(loss=l, entropy=entropy_l)
                            train_losses_epoch.append(l)
                            train_target_entropy_loss.append(entropy_l)
                if self.using_adversarial_training:
                    self.all_tr_losses.append(np.mean(train_losses_epoch))
                    self.all_d_segloss.append(np.mean(train_d_loss))
                    self.train_d_Ours.append(np.mean(train_d_Ours))
                    self.train_d_Kits.append(np.mean(train_d_Kits))
                    self.print_to_log_file(
                        "train loss : %.4f ; seg_loss on D : %.4f" % (
                            self.all_tr_losses[-1], self.all_d_segloss[-1]))
                else:
                    self.all_tr_losses.append(np.mean(train_losses_epoch))
                    self.all_tr_target_entropy_losses.append(np.mean(train_target_entropy_loss))
                    self.print_to_log_file("train loss : %.4f ; target entropy : %.4f" % (
                        self.all_tr_losses[-1], self.all_tr_target_entropy_losses[-1]))
                # validate
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.target_val_gen, None, False, True)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("target loss: %.4f" % self.all_val_losses[-1])
                continue_training = self.on_DyDA_epoch_end(current_DyDA_epoch=x, current_DyDA_iter=ce)
                epoch_end_time = time()
                self.epoch += 1
                self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

    '''
    实验中的对比方法： MinEnt_UDA
    '''
    def MinEnt_UDA(self):
        # _ = self.tr_gen.next()
        # if self.target_gen is not None:
        #     _ = self.target_gen.next()
        # _ = self.target_val_gen.next()

        if torch.cuda.is_available():
            print('debug torch.cuda', torch.cuda.is_available())
            torch.cuda.empty_cache()

        self._maybe_init_amp()
        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\niter: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            train_target_entropy_loss = []
            train_d_loss = []
            train_d_Ours = []
            train_d_Kits = []

            # train one epoch
            self.network.train()

            # if self.use_progress_bar:
            # if self.use_progress_bar:
            with trange(self.num_batches_per_epoch) as tbar:
                for b in tbar:
                    tbar.set_description("iter {}/{}".format(self.epoch + 1, self.max_num_epochs))
                    l, entropy_l = self.run_iteration(self.tr_gen, self.target_gen, True, False)
                    tbar.set_postfix(loss=l, entropy=entropy_l)
                    train_losses_epoch.append(l)
                    train_target_entropy_loss.append(entropy_l)
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.all_tr_target_entropy_losses.append(np.mean(train_target_entropy_loss))
            self.print_to_log_file("train loss : %.4f ; target entropy : %.4f" % (
            self.all_tr_losses[-1], self.all_tr_target_entropy_losses[-1]))
                # self.print_to_log_file("train loss : %.4f " % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.target_val_gen, None, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("target loss: %.4f" % self.all_val_losses[-1])

            # we don't need early stopping
            # self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))


    def ADVENT_UDA(self):
        self.source_label = 0
        self.target_label = 1
        self.all_d_segloss = []
        self.train_d_Ours = []
        self.train_d_Kits = []

        if torch.cuda.is_available():
            print('debug torch.cuda', torch.cuda.is_available())
            torch.cuda.empty_cache()

        self._maybe_init_amp()
        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        if not self.was_initialized:
            self.initialize(True)

        print("AdvEnt Epochs:", self.max_num_epochs, '--------self epoch:', self.epoch)
        self.print_to_log_file('AdvEnt Train epoch:', self.max_num_epochs)
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\niter: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            train_target_entropy_loss = []
            train_d_loss = []
            train_d_Ours = []
            train_d_Kits = []

            # train one epoch
            self.network.train()

            with trange(self.num_batches_per_epoch) as tbar:
                for b in tbar:
                    tbar.set_description("iter {}/{}".format(self.epoch + 1, self.max_num_epochs))
                    l, dlossonseg, dlossonOurs, dlossonKits = self.run_iteration(self.tr_gen, self.target_gen, True,
                                                                                 False, True)
                    tbar.set_postfix(loss=l)
                    train_losses_epoch.append(l)
                    train_d_loss.append(dlossonseg)
                    train_d_Ours.append(dlossonOurs)
                    train_d_Kits.append(dlossonKits)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.all_d_segloss.append(np.mean(train_d_loss))
            self.train_d_Ours.append(np.mean(train_d_Ours))
            self.train_d_Kits.append(np.mean(train_d_Kits))
            self.print_to_log_file(
                    "train loss : %.4f ; seg_loss on D : %.4f" % (self.all_tr_losses[-1], self.all_d_segloss[-1]))
                # self.print_to_log_file("train loss : %.4f " % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.target_val_gen, None, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("target loss: %.4f" % self.all_val_losses[-1])

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

    ''''
    UDA训练
    MinEnt_UDA: 使用MinEnt进行UDA
    AdvEnt_UDA: 使用对抗学习进行UDA
    DyDA：动态划分数据集进行UDA
    '''
    def run_training(self):
        self._maybe_init_amp()
        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()
        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\niter: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            train_target_entropy_loss = []
            train_d_loss = []
            train_d_Ours = []
            train_d_Kits = []

            # train one epoch
            self.network.train()

            # if self.use_progress_bar:
            with trange(self.num_batches_per_epoch) as tbar:
                for b in tbar:
                    tbar.set_description("iter {}/{}".format(self.epoch + 1, self.max_num_epochs))

                    if using_adversarial_training:
                        l, dlossonseg, dlossonOurs, dlossonKits = self.run_iteration(self.tr_gen, self.target_gen, True,
                                                                                     False, True)
                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                        train_d_loss.append(dlossonseg)
                        train_d_Ours.append(dlossonOurs)
                        train_d_Kits.append(dlossonKits)
                    else:
                        l, entropy_l = self.run_iteration(self.tr_gen, self.target_gen, True, False)
                        tbar.set_postfix(loss=l, entropy=entropy_l)
                        train_losses_epoch.append(l)
                        train_target_entropy_loss.append(entropy_l)

            if using_adversarial_training:
                self.all_tr_losses.append(np.mean(train_losses_epoch))
                self.all_d_segloss.append(np.mean(train_d_loss))
                self.train_d_Ours.append(np.mean(train_d_Ours))
                self.train_d_Kits.append(np.mean(train_d_Kits))
                self.print_to_log_file(
                    "train loss : %.4f ; seg_loss on D : %.4f" % (self.all_tr_losses[-1], self.all_d_segloss[-1]))
                # self.print_to_log_file("train loss : %.4f " % self.all_tr_losses[-1])
            else:
                self.all_tr_losses.append(np.mean(train_losses_epoch))
                self.all_tr_target_entropy_losses.append(np.mean(train_target_entropy_loss))
                self.print_to_log_file("train loss : %.4f ; target entropy : %.4f" % (
                    self.all_tr_losses[-1], self.all_tr_target_entropy_losses[-1]))
                # self.print_to_log_file("train loss : %.4f " % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.target_val_gen, None, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("target loss: %.4f" % self.all_val_losses[-1])

            # we don't need early stopping
            # self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        if self.DyDA:
            self.DyDA_UDA()
        # from nnunet.run.run_training import using_adversarial_training
        # if self.DyDA:
        #     print("DyDA epoch sum:", self.dynamic_split_epoch, flush=True)
        #     self.print_to_log_file('DCSS epoch :', self.dynamic_split_epoch,'---', 'DCSS round', self.dynamicDA_period)
        #     self.DyDA_UDA()
        # elif using_adversarial_training:
        #     self.ADVENT_UDA()
        # else:
        #     self.MinEnt_UDA()

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        if self.save_final_checkpoint:
            self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))

        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == 0):
            self.print_to_log_file("saving scheduled checkpoint file...")
            # if not self.save_latest_only:
            self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            # self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            # self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                # self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                # self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                # self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    # self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    # self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                # self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        # plot_dynamic_boundary()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        # self.update_eval_criterion_MA()

        # continue_training = self.manage_patience()
        return True

    def on_DyDA_epoch_end(self, current_DyDA_epoch, current_DyDA_iter):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        # if current_DyDA_epoch in [1, 5, 10]:
        #     save_feature()


        self.maybe_update_lr_DyDA()

        self.maybe_save_checkpoint()

        # self.update_eval_criterion_MA()

        # continue_training = self.manage_patience()
        return True

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target


        return l.detach().cpu().numpy()

    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses

