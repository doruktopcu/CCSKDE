# Sequential Keypoint Density Estimator  
## *An Overlooked Baseline of Skeleton-Based Video Anomaly Detection*  
#### **ICCV 2025 • Highlight Paper**

---

This repository provides the **official implementation** of the paper:  
> **"Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection"**  
> by *Anja Delić, Matej Grcić, and Siniša Šegvić.*

---

[![Conference](https://img.shields.io/badge/ICCV-2025-blue.svg)](https://iccv2025.thecvf.com)
[![arXiv](https://img.shields.io/badge/arXiv-2506.18368-b31b1b.svg)](https://arxiv.org/abs/2506.18368)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/adelic99/seeker.svg?style=social)](https://github.com/adelic99/seeker)

---

<!-- ## 🧩 Overview

**SeeKer** (Sequential Keypoint Density Estimator) is a lightweight and efficient baseline for **skeleton-based video anomaly detection**.  
It models the conditional density of each keypoint across time and flags motion patterns with low probability as anomalies.  

Despite its simplicity, **SeeKer** achieves **state-of-the-art performance** on UBnormal and MSAD-HR datasets, and remains highly reproducible and interpretable. -->

<!-- <p align="center">
  <img src="docs/overview.png" width="80%" alt="Model Overview"/>
</p> -->

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/SeeKer.git
cd SeeKer
pip install -r requirements.txt
```

### 2. Training
```bash
python seeker.py --args
```


### 3. Evaluation
```bash
python seeker.py --checkpoint PATH --args
```


## 📊 Results
| Dataset      | AUROC       | Gain over prior SOTA  |
| ------------ | ----------- | --------------------- |
| UBnormal     | **77.9**    | +5.1                  |
| UBnormal-HR  | **78.9**    | +7.4                  |
| MSAD-HR      | **61.1**    | +5.4                  |
| ShanghaiTech | **85.5**    | competitive           |


## 🧠 Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{delic2025sequential,
  title={Sequential keypoint density estimator: an overlooked baseline of skeleton-based video anomaly detection},
  author={Deli{\'c}, Anja and Gr{\v{c}}i{\'c}, Matej and {\v{S}}egvi{\'c}, Sini{\v{s}}a},
  journal={arXiv preprint arXiv:2506.18368},
  year={2025}
}
```

## 🤝 Acknowledgments
This implementation is built upon the codebase from [STG-NF](https://github.com/orhir/STG-NF) (Orhir et al.).


## 📄 License
This project is released under the MIT License.
