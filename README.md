# Effective Abstract Reasoning with Dual-Contrast Network
Tao Zhuo, Mohan Kankanhalli

This code is the implementation of our ICLR 2021 [paper](https://openreview.net/pdf?id=ldxlzGYWDmW)

# Results

Average testing accuracy of different models on RAVEN and neutral regime of PGM dataset. Aux means auxiliary annotations.
|Method         | Aux    | Avg   | RAVEN  | PGM   | 
|---------------|--------|-------|--------|-------|
| ResNet-18+DRT | &check;| -     |  59.56 |   -   |
| WReN+Aux      | &check;| 55.44 |  33.97 | 76.90 |
| LEN+Aux       | &check;| 70.85 |  59.40 | 82.30 |
| MXGNet+Aux    | &check;|  -    |   -    |**89.60** |
|  ACL          | &check;|  -    | **93.71**|  -  |
| LSTM          |        | 24.44 | 13.07  | 35.80 |
| CNN           |        | 34.99 | 36.97  | 33.00 |
| WReN          |        | 40.10 | 17.94  | 62.60 |
| Wild-ResNet   |        |  -    |  -     | 48.00 |
| ResNet-50     |        | 64.13 | 86.26  | 42.00 |
| MLRN          |        | 55.33 | 12.50  |**98.03** |
| LEN           |        | 70.50 | 72.90  | 68.10 |
| CoPINet       |        | 73.90 | 91.42  | 56.37 |
| MXGNet        |        | 75.31 | 93.91  | 66.70 |
| DCNet-RC      |        | 78.10 | 92.74  | 63.45 |
| DCNet-CC      |        | 47.10 | 36.47  | 57.76 |
| DCNet         |        |**81.08** | **93.58** | 68.57 |

# Citation
@inproceedings{zhuo2021,
    author={Tao Zhuo and Mohan Kankanhalli},
    title={Effective Abstract Reasoning with Dual-Contrast Network},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2021}
}
