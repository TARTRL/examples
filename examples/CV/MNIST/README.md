# TLaunch Examples for MNIST

### Usage

If you launch your job with python (e.g. `python train.py`), 
you have no need to change your code when you want to launch your job on the k8s cluster.
All you need to do is just replacing `python` with `tlaunchrun basic`:
```shell
tlaunchrun basic train.py
```

There is  practical example to train MNIST on the k8s cluster with TLaunch:
```shell
bash run.sh
```