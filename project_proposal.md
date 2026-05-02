# Context-Conditioned Sequential Keypoint Density Estimation for Traffic Hazard Detection

**Doruk Topcu** N25142281  
doruktopcu@hacettepe.edu.tr  

## Motivation
The Sequential Keypoint Density Estimator (SKDE) provides an elegant, lightweight baseline for Video Anomaly Detection (VAD) by modeling normal human motion as a sequential distribution over 2D skeleton keypoints. However, a fundamental limitation of SKDE is its environmental blindness; it processes each skeleton in complete isolation. In real-world surveillance, an anomaly is often defined not just by the motion itself, but by its spatial context—such as walking normally but dangerously close to moving vehicles. This project aims to introduce environmental conditioning directly into the autoregressive architecture of SKDE, transforming it from a purely motion-based model to a context-aware density estimator, while remaining computationally feasible for a single-semester timeline.

## Core Method
This project builds upon the SKDE architecture proposed by Delić et al. [1]. The baseline model factorizes the joint density of a skeleton sequence via an autoregressive masked fully connected network, predicting the mean $\mu_\theta$ and covariance $\Sigma_\theta$ for multivariate normal distributions of future keypoints. Anomalies are flagged when observed keypoints yield a low log-likelihood under these predicted distributions.

## Proposed Model-Level Contributions
To address the requirement for fundamental Computer Vision and model-level modifications, this project alters the input space and mathematical formulation of the baseline SKDE architecture without relying on overly complex dynamic graph generation:

1. **Environment-Conditioned Autoregressive Factorization:** The current baseline predicts the probability of a keypoint given solely past keypoints: $p_\theta(X_{t,n}|X_{t,<n}, X_\Delta)$. I propose injecting an environmental semantic vector $C_t$ into the network. Using YOLOv11 to extract bounding boxes of critical scene elements (e.g., vehicles, bicycles) from standard datasets, $C_t$ will be calculated as a scalar or lightweight vector representing the relative spatial proximity of the pedestrian to these hazards. $C_t$ will be concatenated into the input layer of the masked fully connected model. The network will be retrained to predict $p_\theta(X_{t,n}|X_{t,<n}, X_\Delta, C_t)$, actively shifting the predicted keypoint covariance $\Sigma_\theta$ based on environmental proximity.

2. **Contextual Covariance Penalty:** By expanding the input vector, the modified network will learn a new mapping for the inverse covariance matrix (via Cholesky decomposition $\Sigma^{-1}_\theta = L_\theta L^\top_\theta$). The ablation study will specifically isolate whether the static inclusion of $C_t$ successfully shrinks the predicted covariance—thereby increasing the anomaly score (Mahalanobis distance)—when a normally-moving pedestrian sequence intersects with a high-risk spatial zone.

## Datasets
To avoid the overhead of synthetic generation, training and evaluation will utilize the standard ShanghaiTech Campus Dataset [2]. ShanghaiTech is highly suitable as it natively contains mixed pedestrian and vehicle/bicycle interactions. YOLOv11 will be utilized strictly as a preprocessing step to extract the non-pedestrian bounding boxes necessary to formulate the $C_t$ vector.

## Expected Outcome
The primary deliverable will be a mathematically extended SKDE model capable of contextual density estimation. The project will yield a quantitative ablation study demonstrating the impact of the environmental vector $C_t$ on standard VAD metrics (AUROC and AP) compared to the vanilla ICCV 2025 baseline, specifically evaluating its effectiveness in detecting context-dependent anomalies on the ShanghaiTech dataset.

## References
[1] A. Delić, M. Grcić, and S. Šegvić, "Sequential keypoint density estimator: an overlooked baseline of skeleton-based video anomaly detection," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Oct. 2025, pp. 11579–11589.

[2] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection – A new baseline," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 6536–6545.