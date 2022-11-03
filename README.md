# O2M-UDA: Unsupervised Dynamic Domain Adaptation for One-to-Multiple Medical Image Segmentation

[NOTE!!!]The code will be gradually opened, and be completely opened after this paper is published.

One-to-multiple medical image segmentation aims to directly test a segmentation model
trained with the medical images of a one-domain site on those of a multiple-domain
site, suffering from segmentation performance degradation on multiple domains. It
avoids additional annotations and helps improve the application value of the model.

However, no successful O2M-UDA work has been reported in one-to-multiple medical image segmentation due to its inherent challenges: 

1) Distribution differences among multiple target domains (among-target differences) caused by different scanning equipments.

2) Distribution differences between one source domain and multiple target domains (source-target differences). 

In this paper, we propose an O2M-UDA framework, Dynamic Domain Adaptation (DyDA), for one-to-multiple medical image segmentation which has two innovations: 1) **Dynamic credible sample strategy (DCSS)** dynamically extracts credible samples from the target site and iteratively updates the number of them. 2) **Hybrid uncertainty learning (HUL)** reduces the voxel-level and domain-level uncertainty simultaneously. 

Experiments on two one-to-multiple medical image segmentation tasks have been conducted to prove the performance of our DyDA.

## Papar
This repository provides the official PyTorch and MindSpore implementation of DyDA in the following paper:

**O2M-UDA: Unsupervised Dynamic Domain Adaptation for One-to-Multiple Medical Image Segmentation**
[Ziyue Jiang](https://github.com/ZoeyJiang/DyDA/edit/main/README.md), [Yuting He](http://19951124.academic.site/?lang=en), Shuai Ye, Pengfei Shao, Xiaomei Zhu, Yi Xu, Yang Chen, Jean-Louis Coatrieux, [Shuo Li](http://www.digitalimaginggroup.ca/members/shuo.php), [Guanyu Yang*](https://cse.seu.edu.cn/2019/0103/c23024a257233/page.htm)
Southeast University

## Official implementation
+ [PyTorch/(https://github.com/ZoeyJiang/DyDA/tree/main/PyTorch)
+ [Mindspore/](https://github.com/ZoeyJiang/DyDA/tree/main/MindSpore)

## Acknowledgement
This research was supported by the National Natural Science Foundation under grants (61828101), CAAI-Huawei MindSpore Open Fund and Scientific Research Foundation of Graduate School of Southeast University (YBPY2139). We thank the Big Data Computing Center of Southeast University for providing the facility support on the numerical calculations in this paper. 
