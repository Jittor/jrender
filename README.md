# DROT-JRender(Jittor)
This repository is part of the code for SIGGRAPH Asia 2022 paper [*Differentiable Rendering using RGBXY Derivatives and Optimal Transport*](https://jkxing.github.io/academic/publication/DROT)

## Installation

## Jittor Install
Please follow the [official documents](https://cg.cs.tsinghua.edu.cn/jittor/download/)

## DROT-JRender install
We suggest create a new Anaconda environment.
```
conda create -n DROT python=3.8
#Install PyTorch, for using geomloss
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install geomloss 
```
pip install pykeops
pip install geomloss
```
Then clone this repo

if some error occurs, the following tips may help:
- GCC version should under 10
- Install correct version of nvidia-gl driver(server version may fail)
- CUDA version/Python-dev version should exactly match everywhere

## Demo

Since JRender(Jittor) doesn't support methods such as multi object rendering, we only provide a core implementation and a single object translation demo. 
```
python demo-drl.py
```
