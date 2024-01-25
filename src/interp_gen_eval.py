import os
import sys
import argparse 
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import numpy as np
import random
import json
import pickle as pkl

import config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def evaluate(
    model,tokenizer,
    data_point,
    generation_config
):
    prompt = generate_prompt(data_point)
    #print (prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    #print (input_ids[0])
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip().split('\n')[0]

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:\n""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:\n""" # noqa: E501

def interp_generation(model,tokenizer,args):
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams
    )   
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if args.DATASET=='meme-captions':
        load_path=os.path.join(args.PATH,args.DATASET,
                                'meme-cap/data',
                                'memes-test_up.json')
    elif args.DATASET=='hatred':
        load_path=os.path.join(args.PATH,args.DATASET,
                                'annotations',
                                'fhm_test_reasonings_up.json')
    data=load_json(load_path)
    print ('Number of examples:',len(data))
    print ('\t',load_path)
    invalid=[]
    random.shuffle(data)
    save_path=os.path.join(args.DATASET,args.load_iter+'.pkl')
    vis=0
    total={}
    for i, row in enumerate(data):
        vis+=1
        #result=evaluate(model, tokenizer, row, generation_config)
        #scores.append(score)
        try:
            result=evaluate(model, tokenizer, row, generation_config)
        except:
            print ('Invalid:',row['img'])
            invalid.append(row['img'])
        total[row['img']]=result
        if i%500==0:
            print ('\tAlready finished...',i)
    pkl.dump(total,open(save_path,'wb'))
    return total

if __name__ == "__main__":
    args=config.parse_opt()
    set_seed(args.SEED)

    #model configurations
    load_8bit=args.load_8bit
    base_model=args.base_model
    lora_weights=args.lora_path+args.load_iter
    print ('Loading LoRA from:',lora_weights)
    device="cuda"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    interp_generation(model,tokenizer,args)