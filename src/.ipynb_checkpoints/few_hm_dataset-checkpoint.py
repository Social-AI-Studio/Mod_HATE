import os
import json
import pickle as pkl
import numpy as np
import torch
import random

from hfm_gen_eval import hfm_generation
from transformers import LlamaTokenizer
from datasets import load_dataset

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
    data=json.load(open(path,'rb'))
    return data

def process_data(rows,captions,num_shots):
    random.shuffle(rows)
    instruct="Please decide whether the meme is hateful according to its image caption and meme text."
    label_mapper={
        0:'No',
        1:'Yes'
    }
    counts={0:0,1:0}
    lines=[]
    origs=[]
    for row in rows:
        if counts[0]==num_shots and counts[1]==num_shots:
            break
        img=row['img']
        #cap=meme_captions[img]
        label=row['label']
        if counts[label]==num_shots:
            continue
        counts[label]+=1
        origs.append(row)
        cap=captions[img]
        ans=label_mapper[label]
        meme_text=row['clean_sent']
        """
        words=meme_text.split(' ')
        if len(words)>32:
            words=words[:32]
        words=' '.join(words)
        """
        cur_row={
            'img':img,
            'input':'Image caption:'+cap+'\n'+'Meme text:'+meme_text,
            'output':ans,
            'instruction':instruct
        }
        lines.append(cur_row)
    return lines,origs

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

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:{data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:{data_point["output"]}""" # noqa: E501

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

class Few_HM_Data():
    #mem, off, harm
    #using opt for arguements, rather than args in other files
    def __init__(self,opt,tokenizer,mode='train'):
        super(Few_HM_Data,self).__init__()
        self.opt=opt
        self.mode=mode
        self.instruct="Please decide whether the meme is hateful according to its image caption and meme text."
        random.seed(opt.SEED)
        self.label_mapper={
            0:self.opt.POS_WORD,
            1:self.opt.NEG_WORD}
        print ('Initializing tokenizer from:',self.opt.base_model)
        print ('\tVerbolizer:',self.opt.POS_WORD,self.opt.NEG_WORD)
        self.tokenizer=tokenizer

        self.entries=self.get_dataset()

    def organize_data(self,save_path):
        data_raw=load_json(os.path.join(self.opt.PATH,'domain_splits',self.opt.DATASET+'_'+self.mode+'.json'))
        captions=load_pkl(os.path.join('../BLIP-2/results',self.opt.DATASET+'-generic.pkl'))
        data,origs=process_data(data_raw,captions,self.opt.num_shots)
        print ('Processing files and dumping....')
        json.dump(data,open(save_path,'w'))
        address=[self.mode,str(self.opt.num_shots),str(self.opt.SEED-1110)]
        address='_'.join(address)
        orig_save_path=os.path.join(self.opt.PATH,'domain_splits',
                               self.opt.DATASET+'_'+address+'.json')
        json.dump(origs,open(orig_save_path,'w'))
        

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
        pad_len=self.opt.cutoff_len-len(tokenized_full_prompt["input_ids"])
        #print (tokenized_full_prompt["labels"])
        if pad_len>0:
            tokenized_full_prompt["input_ids"]=[0]*pad_len+tokenized_full_prompt["input_ids"]
            tokenized_full_prompt["labels"]=[-100]*pad_len+tokenized_full_prompt["labels"]
            tokenized_full_prompt["attention_mask"]=[0]*pad_len+tokenized_full_prompt["attention_mask"]
        #print (tokenized_full_prompt["input_ids"])
        #print (tokenized_full_prompt["labels"])
        #print (tokenized_full_prompt["attention_mask"],'\n')
        return tokenized_full_prompt
        
    def get_dataset(self):
        address=[self.mode,str(self.opt.num_shots),str(self.opt.SEED-1110),'up']
        address='_'.join(address)
        load_path=os.path.join(self.opt.PATH,'domain_splits',
                               self.opt.DATASET+'_'+address+'.json')
        if os.path.exists(load_path)==False:
            self.organize_data(load_path)
        entries=load_json(load_path)
        print ('Length of dataset',len(entries))
        print ('Loading data from %s mode, from %s' % (self.mode,load_path))
        #wrap_data=(data['train'].shuffle().map(self.generate_and_tokenize_prompt))
        return entries

    def __getitem__(self,index):
        #query item
        entry=self.entries[index]
        batch=self.generate_and_tokenize_prompt(entry)
        batch["input_ids"]=torch.LongTensor(batch["input_ids"])
        batch["labels"]=torch.LongTensor(batch["labels"])
        batch["attention_mask"]=torch.LongTensor(batch["attention_mask"])
        return batch
        
    def __len__(self):
        return len(self.entries)