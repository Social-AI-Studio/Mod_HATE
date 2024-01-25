import os
import torch
import numpy as np
import transformers
import random
import json
import pickle as pkl

import config
from train import train_for_epochs

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
if __name__ == "__main__":
    args=config.parse_opt()
    set_seed(args.SEED)
    
    device_map = "auto"
    #model initialization
    if args.load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    model = prepare_model_for_int8_training(
        model, 
        use_gradient_checkpointing=args.use_gradient_checkpointing)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    #tokenizer instaintiation
    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    #data preparation
    if args.DATASET in ['mem','harm','mimc']:
        from hm_dataset import HM_Data
        train_cls=HM_Data(args,tokenizer,mode='train')
        val_cls=HM_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['meme-captions','hatred']:
        from gen_dataset import GEN_Data
        train_cls=GEN_Data(args,tokenizer,mode='train')
        val_cls=GEN_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['hate-speech']:
        from text_dataset import TEXT_Data
        train_cls=TEXT_Data(args,tokenizer,mode='train')
        val_cls=TEXT_Data(args,tokenizer,mode='test')
    train_set=train_cls.get_dataset()
    val_set=val_cls.get_dataset()
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    train_for_epochs(model,tokenizer,args,  
                     data_collator,
                     train_set,val_set)