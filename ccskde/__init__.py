"""CCSKDE — Context-Conditioned Sequential Keypoint Density Estimation.

Extension of the SeeKer baseline (Delić et al., ICCV 2025) that injects an
environmental context vector C_t (derived from non-pedestrian object
detections) into the autoregressive density estimator.

The upstream `seeker/` package is treated as a read-only reference; all
modifications live under this package.
"""

__version__ = "0.0.1"
