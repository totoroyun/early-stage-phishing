# Early-stage Phishing Detection on the Ethereum Transaction Network

This is the source code of Soft computing paper [Early-stage phishing detection on the Ethereum transaction network](https://link.springer.com/article/10.1007/s00500-022-07661-0).

![The proposed framework](https://github.com/totoroyun/early-stage-phishing/blob/main/framework.png)

![Time series splitting](https://github.com/totoroyun/early-stage-phishing/blob/main/split.png)
<div align="center">
    <img src="https://github.com/totoroyun/early-stage-phishing/blob/main/split.png" alt="示例图像">
</div>


## Running the Experiments

To reproduce our experiments, follow these steps:

### Step 1: Download the Dataset

Download the original MulDiGraph dataset from https://xblock.pro/#/dataset/13.

### Step 2: Extract Features and Test Method Performance

Run the early-stage run.py script to extract features and test the performance of our method. We have provided the positive and sampled negative nodes for each divided timestamp, as well as the extracted features in the T1 dataset for your convenience. You can directly use our code to extract further features for additional experiments.

## Citing Our Work

If you compare, build on, or use aspects of our framework, please cite the following paper:
```
@article{wan2022early, 
  title={Early-stage phishing detection on the Ethereum transaction network}, 
  author={Wan, Yun and Xiao, Feng and Zhang, Dapeng}, 
  journal={Soft Computing}, 
  pages={1--13}, 
  year={2022}, 
  publisher={Springer} 
}
```
