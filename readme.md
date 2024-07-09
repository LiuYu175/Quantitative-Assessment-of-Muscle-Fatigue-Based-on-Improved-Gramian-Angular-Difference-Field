# Needs Serious Attention !!!


This project is the experimental source code for the paper **"Quantitative Assessment of Muscle Fatigue Based on Improved Gramian Angular Difference Field"**, which has been submitted to IEEE SENSORS JOURNAL (under review).  
For citation, please inform the author at **smaugfire@163.com**.


# 1. Environment Configuration

Install the package with the following command:

```
pip install -r requirements.txt
```

# 2. Generating Datasets

Running `make_datasets.py` in the `dataset` folder will automatically create all datasets.

## a. Pre-processing of data

Preprocessed files are in `Preprocessing`.

## b. Generate all data sets

In the `ALL` folder, it is first named after the number of channels, followed immediately by the image code.

|  | G-GASF | G-GADF | GASF | GADF | MTF | RP |
| --- | --- | --- | --- | --- | --- | --- |
|  A1 | / | / | / | / | / | / |
| A2 | / | / | / | / | / | / |
| A3 | / | / | / | / | / | / |
| Max | / | / | / | / | / | / |

# 3. Training Models

Run `train.py` to train all models and datasets, results are in `output/train`.

# 4. Test Models

Run `test.py` on all training results, the results are in `output/test`.

# Reference Codebase

```
@misc{rw2019timm,
author = {Ross Wightman},
title = {PyTorch Image Models},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
doi = {10.5281/zenodo.4414861},
howpublished = {\url{[https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)}}
}
```