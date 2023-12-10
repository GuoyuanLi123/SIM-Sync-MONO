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

```

## Citation
If you find our work useful in your research, please consider citing:
```BibTeX
@inproceedings{SIM-Sync-Mono,
 title={SIM-Sync-Mono: Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module},
 author={Xihang Yu, Yuchen Zhou, Guoyuan Li},
 year={2023},
 booktitle=IEEE/CVF Conference on Computer Vision and Pattern Recognition
}
```
## License
See the [LICENSE](LICENSE) for more details.

## Issues & Help
For help or issues using Robust CVD, please submit a GitHub issue or a PR request.

Before you do this, make sure you have checked [CODE_OF_CONDUCT](./CODE_OF_CONDUCT.md), [CONTRIBUTING](./CONTRIBUTING.md), [ISSUE_TEMPLATE](docs/.github/ISSUE_TEMPLATE.md), and [PR_TEMPLATE](docs/.github/PR_TEMPLATE.md).

## Acknowledgements
Check our previous work on [Consistent Video Depth Estimation](https://github.com/facebookresearch/consistent_depth).

We also thank the authors for releasing [PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [Ceres Solver](http://ceres-solver.org/), [OpenCV](http://opencv.org/), [Eigen](https://eigen.tuxfamily.org/), [MiDaS](https://github.com/intel-isl/MiDaS), [RAFT](https://github.com/princeton-vl/RAFT), and [detectron2](https://github.com/facebookresearch/detectron2).
