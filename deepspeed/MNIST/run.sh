#!/bin/bash
tlaunchrun --lpjob_name mnist-deepspeed --image docker.4pd.io/tlaunch/tlaunch:0.0.2-cuda11.2-cv --image_pull_policy 'Always' --gpu 4 --gpu_memory 10000 \
        --gpu_cores 100 --set_save_path deepspeed --num_nodes 2 --num_gpus 4 \
        train.py --save-model --deepspeed --deepspeed_config ./ds_config.json
