# P-tuning v2

Source codes adopted from: [ACL 2022] [P-Tuning v2: Prompt Tuning Can Be Comparable to Finetuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602) 

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for P-tuning v2:

```shell
conda create -n pt2 python=3.8.5
conda activate pt2
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Setup For Tlaunch
Move directory "data" and "code" to "/TData"

```shell
cd P-tuning-v2
cp -r code /TData
cp -r data /TData
```

### Training
Run training scripts in [run_script](./run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_conll04_bert.sh
```

## Citation

If you find our work useful, please kindly cite our paper:

```
@article{DBLP:journals/corr/abs-2110-07602,
  author    = {Xiao Liu and
               Kaixuan Ji and
               Yicheng Fu and
               Zhengxiao Du and
               Zhilin Yang and
               Jie Tang},
  title     = {P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally
               Across Scales and Tasks},
  journal   = {CoRR},
  volume    = {abs/2110.07602},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07602},
  eprinttype = {arXiv},
  eprint    = {2110.07602},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-07602.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
