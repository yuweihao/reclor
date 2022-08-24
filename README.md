# ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning (ICLR 2020)

This repository contains PyTorch code for the paper: Weihao Yu*, Zihang Jiang*, Yanfei Dong, and Jiashi Feng, [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://openreview.net/pdf?id=HJgJtT4tvB), ICLR 2020 (* equal contribution). For the data and more information, please check out the [project page](http://whyu.me/reclor). 

## Setting up and using the repo
1. Set up the environment. Install Python3.5+, PyTorch 1.0+, [Transformers](https://github.com/huggingface/transformers) and [apex](https://github.com/NVIDIA/apex). I recommend the [Anaconda distribution](https://www.anaconda.com/distribution/) to set up Python environment. Refer to [pytorch.org](https://pytorch.org/) to install PyTorch. Then install [Transformers package](https://github.com/huggingface/transformers) by
```bash
pip install transformers==2.3.0
```
Then refer to [apex](https://github.com/NVIDIA/apex) to install Nvidia apex for mixed precision training.

2. Clone the repo by 
```bash
git clone https://github.com/yuweihao/reclor.git
```
3. Get the dataset. Download the dataset from this [Download Link](https://github.com/yuweihao/reclor/releases/download/v1/reclor_data.zip). The unzip password is `for_non-commercial_research_purpose_only`. Or you can use the following command to unzip the file:

```
mkdir reclor_data && unzip -P for_non-commercial_research_purpose_only -d reclor_data reclor_data.zip
```

4. Run the scripts in the main directory by such as 
```bash
sh scripts/run_roberta_large.sh
```

5. (Optional) After running the script, you can find `best_dev_results.txt` in the checkpoint directory which record the best result on validation set, and the predicted file `test_preds.npy` for testing set that you can submit to the [EvalAI leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347) to see the accuracy. The predicted file `test_preds.npy` contains [3, 1, 0, 2, ...] where `i`-th element represents the predicted label of i-th question with "id_string": "test_i" in test.json.


## Results

We obtain the following results with `Ubuntu 16.04, NVIDIA driver 430, PyTorch 1.3.1, cudatoolkit 10.1, numpy 1.17.4, NVIDIA apex, and NVIDIA TITAN RTX GPU` (we find the results are different between TITAN RTX and GeForce RTX 2080TI when runing large models).

|  Model   | Val  | Test | Test-E | Test-H |
|  ----  | ----  |  ----  | ----  |  ----  |
|  bert-base  | 54.6  |  47.3 | 71.6 |  28.2  |
|  bert-large  | 53.8  |  49.8  | 72.0  |  32.3  |
|  xlnet-base  | 55.8  |  50.4  | 75.2  |  32.9  |
|  xlnet-large  | 62.0  |  56.0 | 75.7  |  40.5  |
|  roberta-base  | 55.0  |  48.5  | 71.1  |  30.7  |
|  roberta-large  | 62.6  |  55.6  | 75.5  |  40.0  |

If you could not obtain similar performance in your environment and device, maybe you can try different random seeds.

## Bibtex

```
@inproceedings{yu2020reclor,
        author = {Yu, Weihao and Jiang, Zihang and Dong, Yanfei and Feng, Jiashi},
        title = {ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning},
        booktitle = {International Conference on Learning Representations (ICLR)},
        month = {April},
        year = {2020}
}
```

## Acknowledgment
Weihao Yu would like to thank TPU Research Cloud (TRC) program for the support of partial computational resources.
