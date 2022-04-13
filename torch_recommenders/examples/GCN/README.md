Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

This implementation makes use of the Cora dataset from https://linqs.soe.ucsc.edu/data.


<div align=center><img src="https://relational.fit.cvut.cz/assets/img/datasets-generated/CORA.svg" width="50%;" style="float:center"/></div>


## Requirements
	* python==3.8.12
	* pytorch==1.10.2
	* numpy==1.21.2
	* scipy==1.5.4
	* tensorboardX==2.5 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)


## Usage


- GCN
  ```python main.py```
- FCN 
  ```python main.py --model=FCN```

## References

[1] [Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" 2016.](https://arxiv.org/abs/1609.02907)

[2] [tkipf/pygcn: Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)

[3] [Graph Convolutional Networks (Pytorch) | Chioni Blog](https://chioni.github.io/posts/gnn/#graph-neural-network-gnn)

## Cite

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
