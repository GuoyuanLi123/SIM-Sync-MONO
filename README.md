# \[EECS 442 Project\] SIM-Sync-Mono

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sE0VmWCuL6HUad3yXHEZoCvJKzAvINpz?authuser=1#scrollTo=_p7km19VLzx3)

This repository contains a Python implementation of Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module, as described in the paper:

`Xihang Yu, Yuchen Zhou, Guoyuan Li. SIM-Sync-Mono: Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module. EECS 442 Fall 2023`

###  [Paper](https://arxiv.org/pdf/2012.05901.pdf) | [Video](https://www.youtube.com) | [Colab](https://colab.research.google.com/drive/1sE0VmWCuL6HUad3yXHEZoCvJKzAvINpz?authuser=1#scrollTo=_p7km19VLzx3)

We present an algorithm for estimating consistent dense depth maps and camera poses from a monocular video. We integrate a learning-based depth prior, in the form of a convolutional neural network trained for single-image depth estimation, with geometric optimization, to estimate a smooth camera trajectory as well as detailed and stable depth reconstruction.

![bundle_adjustment](https://drive.google.com/uc?export=view&id=10VHFFIzCttl6t5LkRXRZZ0uXv3dI9G1D)


## Installation

Please refer to the colab notebook for how to install the dependencies.

## Demo

Please refer to the colab notebook for how to run the demo.

## Folder Structure

```bash
SIM-Synb-Mono/
    utils/
    TEASER-plusplus/
    SimSyncRegularized.py
    pose_optimization.py
    TEASER_SIM_Sync.py
    pose_optimizer_tum_v2.py
    Dataset/
    visualization_results/

```

## Citation
If you find our work useful in your research, please consider citing:
```BibTeX
@inproceedings{SIM-Sync-Mono,
 title={SIM-Sync-Mono: Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module},
 author={Xihang Yu, Yuchen Zhou, Guoyuan Li},
 year={2023},
}
```

## Acknowledgements
We also thank the authors for releasing [PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [MOSEK Solver](https://www.mosek.com/), [open3d](http://www.open3d.org/), [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus).
