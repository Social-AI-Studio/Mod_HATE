import argparse 
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, 
                        default='set the path')
    parser.add_argument('--DATASET', type=str, 
                        default='mem')
    parser.add_argument('--SEED', type=int, 
                        default=1111)
    parser.add_argument('--SAVE_NUM', type=int, 
                        default=1)
    parser.add_argument('--DEBUG', type=bool, 
                        default=False)
    parser.add_argument('--NEG_WORD', type=str, 
                        default='Yes')
    parser.add_argument('--POS_WORD', type=str, 
                        default='No')

    #new added settings for LoRA
    parser.add_argument('--max_inference_step', type=int, 
                        default=40)
    parser.add_argument('--num_shots', type=int, 
                        default=4)
    parser.add_argument('--lora_dir', type=str, 
                        default='set the path')
    parser.add_argument('--lora_modules', type=str, 
                        default='hate-exp,meme-captions,hate-speech')

    #fine-tuning LLaMA related
    parser.add_argument('--base_model', type=str, 
                        default='yahma/llama-7b-hf')
    parser.add_argument('--load_8bit', type=bool, 
                        default=False)
    parser.add_argument('--batch_size', type=int, 
                        default=16)
    parser.add_argument('--micro_batch_size', type=int, 
                        default=4)
    parser.add_argument('--logging_steps', type=int, 
                        default=5)
    parser.add_argument('--save_total_limit', type=int, 
                        default=5)
    parser.add_argument('--warmup_steps', type=int, 
                        default=100)
    parser.add_argument('--num_epochs', type=int, 
                        default=2)
    parser.add_argument('--learning_rate', type=float, 
                        default=5e-4)
    parser.add_argument('--cutoff_len', type=int, help='Maximum sequence length to process.',
                        default=256)
    parser.add_argument('--use_gradient_checkpointing', type=bool, 
                        default=False)
    parser.add_argument('--group_by_length', type=bool, 
                        default=False)
    parser.add_argument('--fp16', type=bool, 
                        default=True)

    parser.add_argument('--lora_r', type=int, default=8,
                        help='curvature.')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='The initialization coefficient of lora-alpha.')  
    parser.add_argument('--lora_dropout', type=int, default=0.05,
                        help='The initialization coefficient of lora_dropout.')
    parser.add_argument('--target_modules', type=str, 
                        default=r'.*language_model.*\.(q_proj|v_proj)')
    parser.add_argument('--train_on_inputs', type=bool, 
                        default=False)

    #saving and evaluation
    parser.add_argument('--eval_step', type=int, 
                        default=50)
    parser.add_argument('--gen_step', type=int, 
                        default=200)
    parser.add_argument('--gen_start', type=int, 
                        default=350)
    parser.add_argument('--val_set_size', type=int, 
                        default=500)
    parser.add_argument('--save_step', type=int, 
                        default=100)
    parser.add_argument('--output_dir', type=str, 
                        default='set the path')
    parser.add_argument('--resume_from_checkpoint', type=str, 
                         default=None)

    #generation related
    parser.add_argument('--temperature', type=float, 
                        default=0.5)
    parser.add_argument('--top_p', type=float, 
                        default=0.75)
    parser.add_argument('--top_k', type=int, 
                        default=40)
    parser.add_argument('--num_beams', type=int, 
                        default=4)
    parser.add_argument('--max_new_tokens', type=int, 
                        default=128)
    args=parser.parse_args()
    return args
