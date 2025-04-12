# Real Time Domain Adaptation in Semantic Segmentation
Implementation and comparison of several deep learning methods and Convolutional Neural Networks to tackle the problem of domain adaptation in real-time semantic segmentation.

## Authors
- [Vincenzo Avantaggiato](https://github.com/VincenzoAvantaggiato)
- [Michele Cazzola](https://github.com/MicheleCazzola)
- [Marco De Luca](https://github.com/markdeluk)

## General information
**Course**: `Advanced Machine Learning` (`Polytechnic of Turin`).  
**Academic year**: 2024-25, developed from December 2024 to January 2025.  
**Teachers**: Tatiana Tommasi, Claudia Cuttano.  
**Topic**: implementation and comparison of the following models:
- `DeepLabV2`
- `PidNet`
- Bilateral Segmentation Network (`BiSeNet`)
- Short-Term Dense Concatenate (`STDC`) network

using the following techniques to mitigate domain shift:
- data augmentation
- Adversarial Domain Adaptation (`ADDA`)
- image-to-image translation (Domain Adaptation via Cross-Domain Mixed Sampling, `DACS`).

## Repository structure

The repository is structured as follows:
- `models/`: definition of the models
- `losses/`: definition of the losses (CrossEntropy, Bondary, Focal, OHEM)
- `main.ipynb`: implementation of training, domain adaptation strategies and extensions
- `results.ipynb`: performance computed in the various steps of the process
- `report.pdf`: final report of the project, in PDF format

## Dataset
The dataset used is LoveDA [[PDF](https://arxiv.org/pdf/2110.08733)], natively built for domain adaptation tasks in semantic segmentation. Indeed, it contains several images (with the related masks) divided into `urban` and `rural` domain.

## Details
The project is structured in several parts:
- Performance comparison of DeepLabV2 and PidNet in single-domain (`urban`) setting
- Training of PidNet in domain-shift setting (`urban` to `rural`)
- Mitigation of domain shift using data augmentation on PidNet
- Implementation of Adversarial Domain Adaptation and DACS on PidNet, to mitigate domain shift: these approaches, never tried before on PidNet and LoveDA (as January 2025), have been proven unsuccessful in this context
- Comparison of PidNet with BiSeNet and STDC in single-domain and domain-shift approach
- Extension to more sophisticate losses: `OHEM Cross Entropy Loss` and `Focal Loss` 


We summarized our work and showed our results in the project report [[PDF](report.pdf)]

## Main References
**Semantic Segmentation**: "A Brief Survey on Semantic Segmentation with Deep Learning", Shijie Hao, Yuan Zhou, Yanrong Guo [[PDF](https://arxiv.org/abs/1912.10230)] 

**LoveDA dataset**: "LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation", Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, Yanfei Zhong [[PDF](https://arxiv.org/abs/2110.08733)]

**DeepLab**: "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFS", Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille [[PDF](https://arxiv.org/pdf/1606.00915.pdf)]

**PidNet**: "PidNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers", Jiacong Xu, Zixiang Xiong, Shankar P. Bhattacharyya [[PDF](https://arxiv.org/abs/2206.02066)]

**Adversarial Domain Adaptation**: "Learning to Adapt Structured Output Space for Semantic Segmentation", Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, Manmohan Chandraker [[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf)]

**DACS**: "DACS: Domain Adaptation via Cross-domain Mixed Sampling", Wilhelm Tranheden, Viktor Olsson, Juliano Pinto, Lennart Svensson [[PDF](https://arxiv.org/pdf/2007.08702.pdf)]

**BiSeNet**: "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation", Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang [[PDF](https://arxiv.org/pdf/1808.00897.pdf)]

**STDC**: "Rethinking BiSeNet For Real-time Semantic Segmentation", Mingyuan Fan, Shenqi Lai, Junshi Huang, Xiaoming Wei, Zhenhua Chai, Junfeng Luo, Xiaolin Wei [[PDF](https://arxiv.org/abs/2104.13188)]
