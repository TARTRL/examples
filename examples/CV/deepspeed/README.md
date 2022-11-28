# Deepspeed on k8s

[DeepSpeed](https://github.com/microsoft/DeepSpeed) is a deep learning optimization framework to
enable unprecedented scale and speed. We have implemented a TLaunch version of DeepSpeed, which enables
DeepSpeed to run on k8s clusters.

### Usage

If you launch your job with deepspeed (e.g. `deepspeed train.py`), 
you have no need to change your code when you want to use deepspeed on the k8s cluster.
All you need to do is just placing `tlaunchrun` before `deepspeed`:
```shell
tlaunchrun deepspeed train.py
```

### Examples
- [mnist with deepspeed](./MNIST/)
