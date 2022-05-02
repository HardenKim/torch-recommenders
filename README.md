# torch-recommenders
Recommendation System Models by Pytorch


## Models

|Model|Paper|
|------|---|
|Factorization Machine|[S Rendle, "Factorization Machines", 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)|
|Wide&Deep|[HT Cheng, et al. "Wide & Deep Learning for Recommender Systems", 2016.](https://arxiv.org/abs/1606.07792)|
|Neural Collaborative Filtering|[He, Xiangnan, et al. "Neural collaborative filtering", 2017.](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=oEkUs-uK75EAAAAA:UAomJ1kzS9s3Mo8tTg7eoOmZo713fSxhr1wdX9i56MHZ-foO1WfEfHIkCVcw_T464oERdPbFm6sJdPs)|
|Neural Factorization Machine|[X He and TS Chua. "Neural Factorization Machines for Sparse Predictive Analytics", 2017.](https://arxiv.org/abs/1708.05027)|
|DeepFM|[H Guo, et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", 2017.](https://arxiv.org/abs/1703.04247)|
|xDeepFM|[J Lian, et al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems", 2018.](https://arxiv.org/abs/1803.05170)|
|GCN|[Kipf & Welling. "Semi-Supervised Classification with Graph Convolutional Networks", 2016.](https://arxiv.org/abs/1609.02907)|
|SASRec|[W. Kang and J. McAuley, "Self-Attentive Sequential Recommendation," 2018.](https://arxiv.org/abs/1808.09781)|


## Model Comparison

MovieLens(ml-1m) is used as the dataset for model comparision.
To evaluate the performance of item recommendation, I adopted the `leave-one-out` evaluation, which has been widely used in many literatures (NCF, Wide&Deep, SASRec).
Use k=10 (top 10 recommendations) for ranking metrics.
The Hyperparameters for each model are in [`model.ini`](https://github.com/HardenKim/torch-recommenders/blob/master/torch_recommenders/config/model.ini).

### MovieLens-1m

| Model     | mAP@k | nDCG@k | HR@k  |
|-----------|-------|--------|-------|
| FM        | 0.316 | 0.402  | 0.684 |
| Wide&Deep | 0.272 | 0.351  | 0.612 |
| NCF       | 0.312 | 0.396  | 0.667 |
| NFM       | 0.304 | 0.390  | 0.671 |
| DeepFM    | 0.317 | 0.402  | 0.678 |
| xDeepFM   | 0.310 | 0.395  | 0.669 |
| SASRec    | 0.478 | 0.554  | 0.797 |


### KMRD-2m

| Model     | mAP@k | nDCG@k | HR@k  |
|-----------|-------|--------|-------|
| SASRec    | 0.757 | 0.805  | 0.955 |

## Examples

- FM
  ```python main.py --model=fm --dataset=ml-1m --top_k=10```
- Wide & Deep
  ```python main.py --model=wd --dataset=ml-1m --top_k=10```
- NCF
  ```python main.py --model=ncf --dataset=ml-1m --top_k=10```
- NFM
  ```python main.py --model=nfm --dataset=ml-1m --top_k=10```
- DeepFM
  ```python main.py --model=dfm --dataset=ml-1m --top_k=10```
- xDeepFm
  ```python main.py --model=xdfm --dataset=ml-1m --top_k=10```
- SASRec
  ```python main_sasrec.py --model=sasrec --dataset=ml-1m --top_k=10```



## Requirements
	* python>=3.8.12
	* pytorch>=1.10.2
	* numpy>=1.21.2
	* pandas>=1.3.5
	* scipy>=1.5.4
	* tensorboardX>=2.5 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)


## References
- [rixwew/pytorch-fm](https://github.com/rixwew/pytorch-fm)
- [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)
- [pyy0715/Neural-Collaborative-Filtering](https://github.com/pyy0715/Neural-Collaborative-Filtering)