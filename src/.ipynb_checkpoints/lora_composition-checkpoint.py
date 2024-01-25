import os
import torch
import numpy as np
import random
import json
import pickle as pkl

import lora_config
from lora_learning import lorahub_learning

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_json(path):
    data=json.load(open(path,'rb'))
    return data

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
if __name__ == "__main__":
    args=lora_config.parse_opt()
    set_seed(args.SEED)

    #preparing examples: inputs and outputs
    lora_module_list=[]# or you can use none to initialize the list
    modules=args.lora_modules.split(',')
    trained_modules=[os.path.join(args.lora_dir,'LoRA',module) for module in modules]
    lora_module_list.extend(trained_modules)
    lorahub_learning(
        lora_module_list,
        args,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size)