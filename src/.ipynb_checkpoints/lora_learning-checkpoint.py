import os
import torch
import numpy as np
import random
import json
import pickle as pkl
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
import nevergrad as ng
import transformers
from transformers import default_data_collator
import config
import utils
from few_hm_dataset import Few_HM_Data
from hfm_gen_eval import hfm_generation
import copy

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
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

from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,PeftConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
#from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

def load_base_model_and_lora_modules(args,
                                     lora_module_list,
                                     model_name_or_path):
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_peft_model_id = lora_module_list[0]
        
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    # 0 is the default model
    try:
        print ('Loading...',default_peft_model_id)
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id,
                                               torch_dtype=torch.float16)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    #peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in lora_module_list:
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id,
                                                   torch_dtype=torch.float16)
        cache[peft_model_id] = get_peft_model_state_dict(cur_peft_model)

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
    return peft_model, tokenizer, cache

def get_score(weights, tokenizer, model, cache, example_dataset, batch_size, get_loss, get_regular, logger):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = list(cache[lora_module_list[0]].keys())
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
        #print (peft_model_id,lora_state_dict[keys[-1]])
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(tokenizer,example_dataset, model, batch_size, logger)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    #metric_val=loss
    return metric_val

def default_get_loss(tokenizer,example_dataset,model, batch_size,logger):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    #print(len(example_dataset))
    data_batch_size =  batch_size
    # use gpu if available
    #data_collator=transformers.DataCollatorForSeq2Seq(
    #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #)
    train_dataloader = DataLoader(example_dataset,
                                  batch_size,
                                  shuffle=True)
    #print(len(train_dataloader))
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for iter, batch in enumerate(train_dataloader):
            #print (batch)
            #batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch.to(device))
            loss = outputs.loss
            #print(outputs.logits.shape)
            logger.write('%d-th iteration, loss %.2f' % (iter,loss))
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset)

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

def lorahub_learning(lora_module_list, 
                     args,
                     max_inference_step,
                     model_name_or_path=None,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     seed=42):
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    log_path='LoRA'
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    logger=utils.Logger(os.path.join(log_path,args.DATASET+'_'+str(args.SAVE_NUM)+'.txt'))  
    log_hyperpara(logger,args)
    
    #lora_module_list.append('haonan-li/bactrian-ru-llama-7b-lora')
    #lora_module_list.append('haonan-li/bactrian-cs-llama-7b-lora')
    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None
    logger.write('Number of module list in total: %d' % (number_of_loras))
    for i,module_name in enumerate(lora_module_list):
        logger.write('\tThe %d-th module: %s' % (i,module_name))
        
    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(args,
                                                               lora_module_list, 
                                                               model_name_or_path)
    model_copy = copy.deepcopy(model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # process dataset
    dataset = Few_HM_Data(args,tokenizer)
    get_score_partial = partial(get_score, tokenizer=tokenizer,
                                model=model_copy, 
                                cache=cache,
                                example_dataset=dataset,
                                batch_size=batch_size,
                                get_loss=get_loss, 
                                get_regular=get_regular,
                                logger=logger)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    #lora_module_list.pop(-1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model_copy, final_lora)
    model_copy = model_copy.merge_and_unload()
    for i,name in enumerate(lora_module_list):
        logger.write('Module name: %s, module weight %f' % (name,recommendation.value[i]))
    #print (recommendation.value)
    invalid,auc,acc=hfm_generation(model_copy,tokenizer,args)
    if len(invalid)>0:
        logger.write('\tThere are %d invalid examples' % (len(invalid)))
    logger.write('AUC %.2f, Acc %.2f' % (auc,acc))
    return recommendation.value, model, tokenizer
