import os
import json
import pickle as pkl
import numpy as np
import torch
import random

from transformers import LlamaTokenizer
from datasets import load_dataset

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_json(path):
    data=json.load(open(path,'rb'))
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

def process_data(rows,meme_texts,instruct,dataset):
    if dataset=='hatred':
        captions=load_pkl(os.path.join('../BLIP-2/results','mem-generic.pkl'))
    lines=[]
    for row in rows:
        #cap=meme_captions[img]
        if dataset=='meme-captions':
            img=row['img_fname']
            meme_text=meme_texts[img]
            cap=row['img_captions'][0]
            ans=row['meme_captions'][0]
        elif dataset=='hatred':
            img=row['img']
            meme_text=meme_texts[img]
            ans=row['reasonings']
            ans=' It '.join(ans)
            ans='It '+ans
            cap=captions[img]
        cur_row={
            'img':img,
            'input':'Image caption:'+cap+'\n'+'Meme text:'+meme_text,
            'output':ans,
            'instruction':instruct
        }
        lines.append(cur_row)
    return lines

def generate_prompt_test(data_point):
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
    
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501

def tokenize(prompt,tokenizer, cutoff_len,base_model,add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in base_model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in base_model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result

class GEN_Data():
    #mem, off, harm
    #using opt for arguements, rather than args in other files
    def __init__(self,opt,tokenizer,mode='train'):
        super(GEN_Data,self).__init__()
        self.opt=opt
        self.mode=mode
        instructions={
            'meme-captions':"Please interprete the meme according to its image caption and meme text.",
            'hatred':"Please explain the rearson that the meme is hateful given the image caption and meme text"
        }
        self.instruct=instructions[self.opt.DATASET]
        
        print ('Initializing tokenizer from:',self.opt.base_model)
        print ('\tInstruction:',self.instruct)
        self.tokenizer=tokenizer

    def organize_data(self,save_path):
        if self.opt.DATASET=='meme-captions':
            load_path=os.path.join(self.opt.PATH,self.opt.DATASET,
                                   'meme-cap/data',
                                   'memes-'+self.mode+'.json')
            data_raw=read_json(load_path)
            meme_texts=load_pkl(os.path.join(self.opt.PATH,'meme-captions/meme_texts.pkl'))
        elif self.opt.DATASET=='hatred':
            load_path=os.path.join(self.opt.PATH,self.opt.DATASET,
                                   'annotations',
                                   'fhm_'+self.mode+'_reasonings.jsonl')
            data_raw=read_jsonl(load_path)
            meme_texts=load_pkl(os.path.join(self.opt.PATH,'multimodal-hate/mem/meme_texts.pkl'))
        data=process_data(data_raw,meme_texts,self.instruct,self.opt.DATASET)
        print ('Processing files and dumping....')
        json.dump(data,open(save_path,'w'))

    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt,self.tokenizer, 
                                         self.opt.cutoff_len,self.opt.base_model)
        if not self.opt.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, self.tokenizer,
                                             self.opt.cutoff_len,self.opt.base_model, 
                                             add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
        
            tokenized_full_prompt["labels"] = [
                                                  -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt
        
    def get_dataset(self):
        if self.opt.DATASET=='meme-captions':
            load_path=os.path.join(self.opt.PATH,self.opt.DATASET,
                                   'meme-cap/data',
                                   'memes-'+self.mode+'_up.json')
        elif self.opt.DATASET=='hatred':
            load_path=os.path.join(self.opt.PATH,self.opt.DATASET,
                                   'annotations',
                                   'fhm_'+self.mode+'_reasonings_up.json')
        if os.path.exists(load_path)==False:
            self.organize_data(load_path)
        data=load_dataset("json",data_files=load_path)
        print ('Loading data from %s mode, from %s' % (self.mode,load_path))
        if self.opt.DEBUG:
            print ('Select a subset of data...')
            data["train"]=data["train"].select([i for i in range(128)])
        wrap_data=(data['train'].shuffle().map(self.generate_and_tokenize_prompt))
        return wrap_data
    