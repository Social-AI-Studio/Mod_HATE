import os
import sys
import argparse 
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

import random
import json
import pickle as pkl
import lora_config
from sklearn.metrics import roc_auc_score

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.numpy()
    label=label.numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc
    
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
    #info=list (generation_output.keys())
    #print (generation_output['scores'][0].shape)
    #yes: 4874; no: 694
    #Yes: 8241; No: 3782
    #Yes</s>: 3869; No</s>: 1939
    s = generation_output.sequences[0]
    #print (len(generation_output['scores']))
    yes_score=generation_output['scores'][0][0,8241].item()
    no_score=generation_output['scores'][0][0,3782].item()
    if yes_score<-1e10:
        yes_score=-1e10
    if no_score<-1e10:
        no_score=-1e10
    scores=[no_score,yes_score]
    #print (generation_output['scores'],generation_output['sequences_scores'])
    #print (s)
    output = tokenizer.decode(s)
    #print (output)
    return output.split("### Response:")[1].strip().split('\n')[0],scores

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:""" # noqa: E501

def compute_acc(scores,labels):
    idx=torch.argmax(scores,dim=-1)
    labels=labels.squeeze()
    results=sum((idx==labels).int())
    return results

def main(args):
    load_8bit=args.load_8bit
    base_model=args.base_model
    lora_weights=args.lora_dir+'/LoRA/'+args.lora_modules
    print ('Loading LoRA from:',lora_weights)
    device="cuda"
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    total={}
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    #tokenizer.add_eos_token=True
    generation_config = GenerationConfig()
    if device == "cuda":
        print ('Loading models to GPU...')
        model = AutoModelForCausalLM.from_pretrained(
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
        
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    data_path=os.path.join(args.PATH,'domain_splits',args.DATASET+'_test.json')
    data_raw=load_json(data_path)
    #random.shuffle(data_raw)
    #data_raw=data_raw[:50]
    data=[]
    instruct="Please decide whether the meme is hateful according to its image caption and meme text."
    label_mapper={
        0:'no',
        1:'yes'
    }
    captions=load_pkl('../BLIP-2/results/'+args.DATASET+'-generic.pkl')
    for row in data_raw:
        img=row['img']
        label=row['label']
        cap=captions[img]
        meme_text=row['clean_sent']
        ans=label_mapper[label]
        data.append({
            'img':img,
            'label':label,
            'input':'Image caption:'+cap+'\n'+'Meme text:'+meme_text,
            'instruction':instruct,
            'output':ans
        })
    print ('Number of examples:',len(data))
    print ('\t',data_path)
    invalid=[]
    scores=[]
    labels=[]
    random.shuffle(data)
    vis=0
    for i, row in enumerate(data):
        labels.append([row['label']])
        vis+=1
        try:
            result,score=evaluate(model, tokenizer, row, generation_config)
            scores.append(score)
        except:
            print ('Invalid:',row['img'])
            invalid.append(row['img'])
        total[row['img']]={
            'score':score,
            'text':result
        }
    scores=torch.Tensor(scores)
    labels=torch.Tensor(labels).view(-1,1).float()
    #print (scores)
    acc=compute_acc(scores,labels)
    scores=torch.softmax(scores,dim=-1)[:,1].unsqueeze(-1)
    auc=compute_auc_score(scores,labels)
    addr=args.lora_modules.split(',')
    addr.append(args.DATASET)
    addr.append(str(args.num_shots))
    pkl.dump(total,open(os.path.join('results','_'.join(addr)+'.pkl'),'wb'))
    print (invalid)
    print (auc*100.0/vis)
    print (acc*100.0/vis)

if __name__ == "__main__":
    args=lora_config.parse_opt()
    main(args)
