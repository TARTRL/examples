
import os
import sys
import logging
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *


import subprocess  
import xmltodict  
from xml.parsers import expat  
import launchpad as lp

from tlaunch import lp_k8s  
from tlaunch.lp_k8s import Config, Container, Resource  
from tlaunch.lp_k8s.util import get_namespace  

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def install(name):  
    subprocess.call(['pip', 'install', name])  

def get_gpu_status(gpu):  
    gpu_id = gpu['minor_number']  
    product_name = gpu['product_name']  
    memory_total = int(gpu['fb_memory_usage']['total'].split(' ')[0])  
    memory_used = int(gpu['fb_memory_usage']['used'].split(' ')[0])  
    memory_free = int(gpu['fb_memory_usage']['free'].split(' ')[0])  

    return 'GPU:{}\t{}Mb/{}Mb\t{}'.format(gpu_id, memory_used, memory_total, product_name)  

def get_gpus():  
    log = logging.getLogger()  
    log.setLevel(logging.DEBUG)  

    cmd = 'nvidia-smi -x -q'  
    output = subprocess.getoutput(cmd)

    json = xmltodict.parse(output, expat=expat)  
    gpus = json['nvidia_smi_log']['gpu']  

    gpu_status = []  
    if type(gpus) is list:  
        for gpu in gpus:  
            gpu_status.append(get_gpu_status(gpu))  
    elif type(gpus) is dict:  
        gpu_status.append(get_gpu_status(gpus))  

    return {'localhost': gpu_status}  


class TestTrainer:
    def __init__(self, args, logger, resume_from_checkpoint=None, last_checkpoint=None):
        self.args = args
        self.logger = logger
        self.resume_from_checkpoint = resume_from_checkpoint
        self.last_checkpoint = last_checkpoint
    
    def run(self):
        # gpu_status = get_gpus()
        # for host in gpu_status:
        #     logging.getLogger().warning('Host {}:'.format(host))
        #     for g_s in gpu_status[host]:
        #         logging.getLogger().warning(g_s)

        trainer, _ = get_trainer(self.args)
        self.train(trainer, self.resume_from_checkpoint, self.last_checkpoint)
        # lp_k8s.stop()

    def train(self, trainer, resume_from_checkpoint=None, last_checkpoint=None):
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        trainer.log_best_metrics()

if __name__ == '__main__':

    args = get_args()

    _, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer
    
    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    # trainer, predict_dataset = get_trainer(args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    program = lp.Program('ptuningv2')
    node = lp_k8s.CourierNode(
        TestTrainer,
        logger=logger,
        args=args,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        last_checkpoint=last_checkpoint)
    program.add_node(node, label='tester')
    # ns = get_namespace() 
    ns = "tpod-yicheng"
    # command = ['bash', '-c' , 'export LIBCUDA_LOG_LEVEL=0; pip install xmltodict; python3 -u -mtlaunch.lp_k8s.process_entry']  
    config = Config(namespace=ns,  
                    container=Container(namespace=ns,  
                                        # command=command,  
                                        # flags=argv,  
                                        resources=Resource(nvidia_gpu=1,  
                                                           nvidia_gpu_memory=4000,  
                                                           nvidia_gpu_cores=100)))  

    print('start launching')
    lp_k8s.launch(program,  
                  namespace=ns,  
                  group_config={'tester': config})  

