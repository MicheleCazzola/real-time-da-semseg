# Real Time Domain Adaptation in Semantic Segmentation

**Authors**:
- [Michele Cazzola](https://github.com/MicheleCazzola): original and revised project
- [Vincenzo Avantaggiato](https://github.com/VincenzoAvantaggiato): original project
- [Marco De Luca](https://github.com/markdeluk): original project

This original version of the project was developed within the course *Advanced Machine Learning* (Prof. Tatiana Tommasi, TA Claudia Cuttano), at Politecnico di Torino in the A.Y. 2024-25. The original version of the project is available [here](https://github.com/MicheleCazzola/real-time-da-semseg/releases/tag/v1.0).

## Overview

This repository contains the code for the project "Real Time Domain Adaptation in Semantic Segmentation", which is a refactor of the original project developed for the course *Advanced Machine Learning* at Politecnico di Torino.

We studied low-latency domain adaptation techniques for real-time semantic segmentation, focusing mainly on the PIDNet architecture and Adversarial Domain Adaptation (ADDA). Our benchmark is the LoveDA dataset, which contains images from two different domains (urban and rural) and is natively built for domain adaptation tasks in semantic segmentation.

As domain adaptation strategies are often investigated on top of larger architectures (e.g. DeepLabV2), we applied them to PIDNet, proving their effectiveness in this context.

Further updates on self-training techniques and extensions will be available soon.

## Installation

This work is developed using:
- `python=3.11`
- `pytorch=2.11`
- `torchvision=0.26`
- `CUDA=12.8`

To install the required dependencies, run:
```
pip install -r requirements.txt
```

To download the LoveDA dataset, please follow the instructions in the [LoveDA repository](https://github.com/Junjue-Wang/LoveDA).

The pretrained checkpoints for fine-tuning and domain adaptation are available in the repositories of the related works:
- [PIDNet](https://github.com/XuJiacong/PIDNet)
- [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg)
- [BiSeNetV1](https://github.com/CoinCheung/BiSeNet)
- [DeepLabV2](https://github.com/rulixiang/deeplab-pytorch)

Once the setup is finished, the repository should have the following structure:

```
real-time-da-semseg/
  ├── /path/to/data/        <- LoveDA dataset 
  ├── /path/to/models/      <- Models and/or checkpoints
  ├── assets/               <- Pictures used in the final report
  ├── configs/              <- Configuration files
  ├── src/
    ├── dataset/            <- Dataset loading and processing
    ├── losses/             <- Loss functions
    ├── metrics/            <- Segmentation and resource metrics
    ├── models/             <- Segmentation and DA models
    ├── train/              <- Training and evaluation logic
    ├── utils/              <- Utilities
  ├── main.py               <- Entry point
  ├── report.pdf            <- Final report/paper
  ├── [other files]
```

## Run

To perform our same fine-tuning operations, run:
```
python main.py --train \
    --model <MODEL_NAME> \
    --source <SOURCE_DOMAIN> \
    --target <TARGET_DOMAIN> \
    --adaptation adda \
    --checkpoint-path <CHECKPOINT_PATH> \
    --output-dir <OUTPUT_DIR>
```

and set:
- `<MODEL_NAME>` to `pidnet_s`
- `<SOURCE_DOMAIN>` to `urban`
- `<TARGET_DOMAIN>` to `rural`
- `<CHECKPOINT_PATH>` to the path of the pretrained checkpoint for PIDNet
- `<OUTPUT_DIR>` to the path of the directory where the results will be saved

It is possible to train a model without any domain adaptation strategy. In that case, it is sufficient to drop the `--adaptation` argument.
The available models are `deeplab_v2`, `bisenet_v1`, `bisenet_v1_rt`, `stdc1`,  `pidnet_s`.

To evaluate a pretrained model, run:
```
python main.py --evaluate \
    --model <MODEL_NAME> \
    --target <DOMAIN> \
    --pretrained-path <PRETRAINED_MODEL_PATH> \
    --output-dir <OUTPUT_DIR>
```

and set:
- `<PRETRAINED_MODEL_PATH>` to the path of the pretrained checkpoint for PIDNet
- `<OUTPUT_DIR>` to the path of the directory where the results will be saved

To evaluate the resource usage (latency, FPS, parameters, FLOPs), run:
```
python main.py --measure \
    --model <MODEL_NAME> \
    --output-dir <OUTPUT_DIR>
```

and set `<OUTPUT_DIR>` as above.

If you want more control on the run parameters, you can directly edit the [configuration file](./configs/config.yaml). For indications on the available parameters, run:

```
python main.py --help
```

## Acknowledgements
This work relies in part on the following:
- [PIDNet](https://github.com/XuJiacong/PIDNet)
- [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)
- [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [BiSeNetV1](https://github.com/CoinCheung/BiSeNet)
- [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg)
- [DeepLabV2](https://github.com/rulixiang/deeplab-pytorch)