# usage
Follow the following four steps to reproduce the DyDA.

**To better understand the code, you'd better have a basic understanding of nnU-Net.**
The code was implemented based on the nnU-Net. Therefore, in order to better understand the code, it is best to have a basic understanding of how to run the nnU-Net. We assume you have a basic understanding of nnU-Net and we will not introduce more about how to run nnU-Net here.

## preprocess
run **run_preprocess.sh** file in the nnunet/experiment_planning folder.
**!!! NOTE**: organize your dataset following the rules of nnU-Net.

## train the model 
run **run.sh or the run_traing.py** file in the nnunet/run folder.
+ **!!!NOTE**: you should change the params based on your experiments


## inference
run **postProcessing.py** file in the nnunet/mypostprocess folder