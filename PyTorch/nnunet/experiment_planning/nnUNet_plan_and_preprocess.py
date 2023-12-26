
import sys
sys.path.append("..")
sys.path.append("../..")

import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21

using_3d = True
load_pregenerated_planfile = True  
plan_file_p = '/dataset/preprocessed/Task003_mixed_jph_kits/preprocessed_3D/nnUNetPlansv2.1_plans_3D.pkl'


def preprocess_3d(using_3d, json_file_p, p_tr, p_gt, cropped_outpath, preprocess_outpath, plan_path, threads=(8, 8)):
    crop(json_file_p, p_tr, p_gt, True, 8, cropped_outpath)

    dataset_json = load_json(json_file_p)
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer(cropped_outpath, overwrite=False,
                                       num_processes=8)  # this class creates the fingerprint
    _ = dataset_analyzer.analyze_dataset(
        collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

    maybe_mkdir_p(preprocess_outpath)
    shutil.copy(join(cropped_outpath, "dataset_properties.pkl"), preprocess_outpath)
    shutil.copy(json_file_p, preprocess_outpath)

    if using_3d:
        exp_planner = ExperimentPlanner3D_v21(cropped_outpath, preprocess_outpath, plan_path)
        exp_planner.plan_experiment()
        exp_planner.run_preprocessing(threads)
    else:
        exp_planner = ExperimentPlanner2D_v21(cropped_outpath, preprocess_outpath)
        exp_planner.plan_experiment()
        exp_planner.run_preprocessing(threads)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")

    args = parser.parse_args()
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    crop(json_file_p, p_tr, p_gt, False, tf, cropped_outpath)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if using_3d:
        if planner_name3d is not None:
            planner_3d = recursive_find_python_class([search_in], planner_name3d,
                                                     current_module="nnunet.experiment_planning")
            if planner_3d is None:
                raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                                   "nnunet.experiment_planning" % planner_name3d)
        else:
            planner_3d = None
    else:
        if planner_name2d is not None:
            planner_2d = recursive_find_python_class([search_in], planner_name2d,
                                                     current_module="nnunet.experiment_planning")
            if planner_2d is None:
                raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                                   "nnunet.experiment_planning" % planner_name2d)
        else:
            planner_2d = None

    # cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
    # preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
    preprocessing_output_dir_this_task = preprocess_outpath

    # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
    dataset_json = load_json(json_file_p)
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer(cropped_outpath, overwrite=False,
                                       num_processes=tf)  # this class creates the fingerprint
    _ = dataset_analyzer.analyze_dataset(
        collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

    maybe_mkdir_p(preprocess_outpath)
    shutil.copy(join(cropped_outpath, "dataset_properties.pkl"), preprocess_outpath)
    shutil.copy(json_file_p, preprocess_outpath)

    threads = (tl, tf)

    print("number of threads: ", threads, "\n")

    if using_3d:
        exp_planner = planner_3d(cropped_outpath, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)
    else:
        exp_planner = planner_2d(cropped_outpath, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()
