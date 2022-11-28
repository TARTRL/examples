# Train MNIST with torchrun on k8s

[torchrun](https://pytorch.org/docs/stable/elastic/run.html) is currently the recommended distributed launching tool for PyTorch models. 
We have implemented a TLaunch version of torchrun, which enables launch distributed PyTorch models 
on k8s clusters.

### Usage
```shell
bash run.sh
```
