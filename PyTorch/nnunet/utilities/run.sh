#!/bin/bash
#BSUB -J  func         #提交作业名
#BSUB -m gpu20      #提交节点
#BSUB -oo test_output_%J.txt            #结果输出文件
#BSUB -eo test_errput_%J.txt           #错误输出文件
#BSUB -q gpu_v100               #指定提交的队列
#BSUB  -gpu "num=1:mode=exclusive_process:aff=yes"

# 配置anaconda3并激活虚拟环境
export PATH=/seu_share/home/ygy_jzy/anaconda3:$PATH
export PATH=/seu_share/home/ygy_jzy/anaconda3/bin:$PATH
source activate jzy_pytorch

# CUDA环境变量
export LD_LIBRARY_PATH=/seu_share/apps/cuda-10.1/lib64:$LD_LIBRARY_PATH
export INCLUDE=/seu_share/apps/cuda-10.1/include:$INCLUDE
export PATH=/seu_share/apps/cuda-10.1/bin:$PATH
# CUDNN环境变量
export LD_LIBRARY_PATH=/seu_share/home/ygy_jzy/cud/cudnn-7.6.3/lib64/:$LD_LIBRARY_PATH
export INCLUDE=/seu_share/home/ygy_jzy/cud/cudnn-7.6.3/include:$INCLUDE


#配置libjpeg的环境变量
export PATH=$PATH:$jpeg/bin
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$jpeg/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$jpeg/lib
export LIBRARY_PATH=$LIBRARY_PATH:$jpeg/libs
export MANPATH=$MANPATH:$jpeg/man
export PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH LIBRARY_PATH MANPATH


#export nnUNet_raw_data_base="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/raw/raw_data/Task006_mixed_crossModa/"
#export nnUNet_preprocessed="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/preprocessed/Task006_mixed_crossModa/"
#export RESULTS_FOLDER="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/trained_models/Task006_mixed_crossModa/"

export nnUNet_raw_data_base="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/raw/raw_data/Task011_mixed_ACDC_myo_emidec/"
export nnUNet_preprocessed="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/preprocessed/Task011_mixed_ACDC_myo_emidec/"
export RESULTS_FOLDER="/seu_share/home/ygy_jzy/olddata/ygy_jzy/DyDE/dataset/trained_models/Task011_mixed_ACDC_myo_emidec/"

# 检查环境变量
#export CUDA_VISIBLE_DEVICES="1"
export CUDA_LAUNCH_BLOCKING=1  # 便于确定位置


python ./json_gen.py

