# Usage
Follow the following four steps to reproduce the DyDA, among which the Post-processing steps are not necessary.

**To better understand the code, you'd better have a basic understanding of nnU-Net.**
The code was implemented based on the nnU-Net. Therefore, in order to better understand the code, it is best to have a basic understanding of how to run the nnU-Net. We assume you have a basic understanding of nnU-Net and we will not introduce more about how to run nnU-Net here.

## Preprocess
run **run_preprocess.sh** file in the nnunet/experiment_planning folder.
**!!! NOTE**: organize your dataset following the rules of nnU-Net.

## Train the model 
run **run.sh or the run_traing.py** file in the nnunet/run folder.
+ **!!!NOTE**: you should change the params based on your experiments

## Inference
run **run.sh or predict_simple.py** file in the nnuent/inference folder.

## Postprocess
run **postProcessing.py** file in the nnunet/mypostprocess folder