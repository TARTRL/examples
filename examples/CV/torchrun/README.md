# torchrun on k8s

[torchrun](https://pytorch.org/docs/stable/elastic/run.html) is currently the recommended distributed launching tool for PyTorch models. 
We have implemented a TLaunch version of torchrun, which enables launch distributed PyTorch models 
on k8s clusters.

### Usage

If you launch your job with torchrun (e.g. `torchrun train.py`), 
you have no need to change your code when you want to launch the job on the k8s cluster.
All you need to do is just placing `tlaunchrun` before `torchrun`:
```shell
tlaunchrun torchrun train.py
```

### Examples
- [mnist with torchrun](./MNIST/)
