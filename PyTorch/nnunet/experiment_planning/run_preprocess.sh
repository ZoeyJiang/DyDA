export nnUNet_raw_data_base="/dataset/raw/raw_data/Task012_myo_emidec"
export nnUNet_preprocessed="/dataset/preprocessed/Task012_myo_emidec"
export RESULTS_FOLDER="/dataset/trained_models/Task012_myo_emidec"



# 检查环境变量
export CUDA_VISIBLE_DEVICES="1"


#python ./nnUNet_plan_and_preprocess.py -pl2d None  # 强制不用2维方法
conda info -e
conda list
 python ./nnUNet_plan_and_preprocess.py -pl3d None  # 强制不用3维方法

