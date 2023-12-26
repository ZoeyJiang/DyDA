
export nnUNet_raw_data_base="dataset/mixed"
export nnUNet_preprocessed="dataset/preprocess_generated"
export RESULTS_FOLDER="/model"

# 检查环境变量
export CUDA_VISIBLE_DEVICES="1"


python postProcessing.py