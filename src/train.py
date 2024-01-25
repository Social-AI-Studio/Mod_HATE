import os
import torch
import numpy as np
import random
import json
import time
import math
import pickle as pkl
import sys
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import transformers
from transformers.utils import is_sagemaker_mp_enabled, is_torch_tpu_available
from transformers.trainer_utils import ShardedDDPOption,has_length
from transformers.trainer_callback import TrainerState
from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version

from hfm_gen_eval import hfm_generation
from hsp_gen_eval import hsp_generation
import utils

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.numpy()
    label=label.numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc

def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def mine_maybe_log_save_evaluate(trainer,all_iters, all_acc, all_auc, logger, opt,tokenizer,
                                 tr_loss, model, trial, epoch, ignore_keys_for_eval):
    if trainer.control.should_log:
        if is_torch_tpu_available():
            xm.mark_step()
        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = trainer._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss
        logger.write('\tIteration %d' % (trainer.state.global_step))
        logger.write('\tLoss: %.2f, learning rate: %f' %
                     (round(tr_loss_scalar / (trainer.state.global_step - trainer._globalstep_last_logged), 4),
                      trainer._get_learning_rate()))

        trainer._total_loss_scalar += tr_loss_scalar
        trainer._globalstep_last_logged = trainer.state.global_step
        trainer.store_flos()
    metrics = None
    if trainer.control.should_evaluate:
        logger.write('Iteration %d, evaluation...' % (trainer.state.global_step))
        
        if isinstance(trainer.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(ignore_keys=ignore_keys_for_eval)
        trainer._report_to_hp_search(trial, trainer.state.global_step, metrics)
        logger.write('\tEval loss: %.2f, runtime %.2f' % 
                     (metrics['eval_loss'],metrics['eval_runtime']))
        
        if trainer.state.global_step>=opt.gen_start and trainer.state.global_step%opt.gen_step==0 and opt.DATASET in ['mem','harm','mimc']:
            invalid,auc,acc=hfm_generation(model,tokenizer,opt)
            if len(invalid)>0:
                logger.write('\tThere are %d invalid examples' % (len(invalid)))
            logger.write('\tAUC %.2f, Acc %.2f' % (auc,acc))
            all_acc.append(acc)
            all_auc.append(auc)
            all_iters.append(trainer.state.global_step)
        elif trainer.state.global_step>=opt.gen_start and trainer.state.global_step%opt.gen_step==0 and opt.DATASET in ['hate-speech']:
            invalid,auc,acc=hsp_generation(model,tokenizer,opt)
            if len(invalid)>0:
                logger.write('\tThere are %d invalid examples' % (len(invalid)))
            logger.write('\tAUC %.2f, Acc %.2f' % (auc,acc))
            all_acc.append(acc)
            all_auc.append(auc)
            all_iters.append(trainer.state.global_step)
        # Run delayed LR scheduler now that metrics are populated
        if isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_to_check = trainer.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            trainer.lr_scheduler.step(metrics[metric_to_check])
    if trainer.control.should_save:
        trainer._save_checkpoint(model, trial, metrics=metrics)
        trainer.control = trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)

def rewrite_train(trainer,tokenizer,opt,logger,trail=None,ignore_keys_for_eval=None):
    #using opt for my own config to avoid confusion with args in trainer
    trainer._memory_tracker.start()
    args=trainer.args
    trainer.is_in_train = True
    trial=None
    if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
        trainer._move_model_to_device(trainer.model, args.device)
    trainer._hp_search_setup(trial)
    trainer._train_batch_size = trainer.args.train_batch_size
    trainer.accelerator.free_memory()
    #initializing dataloaders
    train_dataloader = trainer.get_train_dataloader()
    total_train_batch_size = trainer._train_batch_size * args.gradient_accumulation_steps * args.world_size
    logger.write('Training batch size: %d' % (trainer._train_batch_size))#bz per device
    logger.write('Total training batch size: %d' % (total_train_batch_size))
    logger.write('Total number of epochs: %d' % (opt.num_epochs))
    #getting the length of data loaders
    len_dataloader = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        #the dataloader is devided with batch size per device
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = trainer.num_examples(train_dataloader)
        logger.write('\tLength of train dataloader: %d' % (len_dataloader))
        logger.write('\tNumber of iterations per epoch: %d' % (num_update_steps_per_epoch))
        logger.write('\tNumber of instances: %d' % (num_examples))
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = trainer.num_examples(train_dataloader) * args.num_train_epochs
        logger.write('\tTotal number of iterations: %d' % (max_steps))
    elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )

    delay_optimizer_creation = (
        trainer.sharded_ddp is not None
        and trainer.sharded_ddp != ShardedDDPOption.SIMPLE
        or is_sagemaker_mp_enabled()
        or trainer.fsdp is not None
        or trainer.is_fsdp_enabled
        )
    if trainer._created_lr_scheduler:
        trainer.lr_scheduler = None
        trainer._created_lr_scheduler = False
    if not delay_optimizer_creation:
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)
        trainer.state = TrainerState()
        trainer.state.is_hyper_param_search = trial is not None
    #about model saving and evaluation
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            trainer.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            trainer.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            trainer.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            trainer.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            trainer.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            trainer.state.save_steps = args.save_steps

    model = trainer._wrap_model(trainer.model_wrapped)
    use_accelerator_prepare = True if model is trainer.model else False
    logger.write('\tAccelerate: %d' % (int(use_accelerator_prepare)))
    if delay_optimizer_creation:
        if use_accelerator_prepare:
            trainer.model = trainer.accelerator.prepare(trainer.model)
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)
    if use_accelerator_prepare:
        trainer.model.train()
        if hasattr(trainer.lr_scheduler, "step"):
            if trainer.use_apex:
                model = trainer.accelerator.prepare(trainer.model)
            else:
                model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                trainer.model, trainer.optimizer, trainer.lr_scheduler
            )
    if model is not trainer.model:
        trainer.model_wrapped = model
    #resume_from_checkpoint is set to be None
    trainer._load_optimizer_and_scheduler(None)

    trainer.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None
    # Update the references
    trainer.callback_handler.model = trainer.model
    trainer.callback_handler.optimizer = trainer.optimizer
    trainer.callback_handler.lr_scheduler = trainer.lr_scheduler
    trainer.callback_handler.train_dataloader = train_dataloader
    #hp and trail both set to be None
    trainer.state.trial_params = None
    trainer.state.max_steps = max_steps
    trainer.state.num_train_epochs = num_train_epochs
    trainer.state.is_local_process_zero = trainer.is_local_process_zero()
    trainer.state.is_world_process_zero = trainer.is_world_process_zero()
    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    trainer._total_loss_scalar = 0.0
    trainer._globalstep_last_logged = trainer.state.global_step
    model.zero_grad()
    trainer.control = trainer.callback_handler.on_train_begin(args, trainer.state, trainer.control)

     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:
        for epoch in range(epochs_trained):
            for _ in train_dataloader:
                break

    total_batched_samples = 0
    all_acc=[]
    all_auc=[]
    all_iters=[]
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader        
        if args.past_index >= 0:
            trainer._past = None
        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        logger.write('Epoch: %d out of %d' % (epoch+1, num_train_epochs))
        logger.write('\tIterations in total %d' % (steps_in_epoch))
        trainer.control = trainer.callback_handler.on_epoch_begin(args, trainer.state, trainer.control)
        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True
        step = -1
        #step_trained_in_current_epoch: resuming from checkpoints
        for step, inputs in enumerate(epoch_iterator):
            #print (inputs)
            total_batched_samples += 1
            if rng_to_sync:
                trainer._load_rng_state(None)
                rng_to_sync = False

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                if steps_trained_in_current_epoch == 0:
                    trainer._load_rng_state(None)
                continue
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None

            if step % args.gradient_accumulation_steps == 0:
                trainer.control = trainer.callback_handler.on_step_begin(args, trainer.state, trainer.control)

            with trainer.accelerator.accumulate(model):
                tr_loss_step = trainer.training_step(model, inputs)

            if (
                args.logging_nan_inf_filter
                and not is_torch_tpu_available()
                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
            ):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + trainer.state.global_step - trainer._globalstep_last_logged)
            else:
                tr_loss += tr_loss_step
                
            trainer.current_flos += float(trainer.floating_point_ops(inputs))
            #steps_in_epoch number of steps for each epoch
            is_last_step_and_steps_less_than_grad_acc = (
                steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
            )
            if (
                total_batched_samples % args.gradient_accumulation_steps == 0
                or
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                is_last_step_and_steps_less_than_grad_acc
            ):
                # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                # in accelerate. So, explicitly enable sync gradients to True in that case.
                if is_last_step_and_steps_less_than_grad_acc or (
                    version.parse(accelerate_version) <= version.parse("0.20.3")
                ):
                    trainer.accelerator.gradient_state._set_sync_gradients(True)
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping
                    if trainer.do_grad_scaling:
                        # Reduce gradients first for XLA
                        if is_torch_tpu_available():
                            gradients = xm._fetch_gradients(trainer.optimizer)
                            xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                        # AMP: gradients need unscaling
                        trainer.scaler.unscale_(trainer.optimizer)
                    if is_sagemaker_mp_enabled() and args.fp16:
                        trainer.optimizer.clip_master_grads(args.max_grad_norm)
                    elif hasattr(trainer.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        trainer.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    elif trainer.use_apex:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(
                            amp.master_params(trainer.optimizer),
                            args.max_grad_norm,
                        )
                    else:
                        trainer.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )
                        
                # Optimizer step
                optimizer_was_run = True
                if is_torch_tpu_available():
                    if trainer.do_grad_scaling:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                        trainer.optimizer.step()
                elif trainer.do_grad_scaling:
                    scale_before = trainer.scaler.get_scale()
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    scale_after = trainer.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    trainer.optimizer.step()
                    optimizer_was_run = not trainer.accelerator.optimizer_step_was_skipped

                if optimizer_was_run:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.lr_scheduler.step()

                model.zero_grad()
                trainer.state.global_step += 1
                trainer.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                trainer.control = trainer.callback_handler.on_step_end(args, trainer.state, trainer.control)
                """
                re-writing to record loss and evaluate
                """
                #original: trainer._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                mine_maybe_log_save_evaluate(trainer,all_iters,all_acc, all_auc,logger,opt,tokenizer,
                                             tr_loss, model, trial, epoch, ignore_keys_for_eval)
            else:
                trainer.control = trainer.callback_handler.on_substep_end(args, trainer.state, trainer.control)
            if trainer.control.should_epoch_stop or trainer.control.should_training_stop:
                break
        if step < 0:
            trainer.control.should_training_stop = True
        trainer.control = trainer.callback_handler.on_epoch_end(args, trainer.state, trainer.control)
        #original: trainer._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        mine_maybe_log_save_evaluate(trainer,all_iters,all_acc, all_auc, logger,opt,tokenizer,
                                     tr_loss, model, trial, epoch, ignore_keys_for_eval)
        if trainer.control.should_training_stop:
            break
    if args.past_index and hasattr(trainer, "_past"):
        # Clean the state at the end of training
        delattr(trainer, "_past")
    logger.write("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if len(all_acc)>0:
        max_idx=sorted(range(len(all_iters)),
                        key=lambda k: all_auc[k]+all_acc[k],
                   reverse=True)[0]
        logger.write('Maximum epoch: %d' %(all_iters[max_idx]))
        logger.write('\tevaluation auc: %.2f, accuracy: %.2f' % (all_auc[max_idx], 
                                                                 all_acc[max_idx]))

def train_for_epochs(model,tokenizer,args,
                     data_collator,
                     train_set,test_set):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    #logger initialization
    log_path=os.path.join(args.DATASET)
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    logger=utils.Logger(os.path.join(log_path,str(args.SAVE_NUM)+'.txt'))  
    log_hyperpara(logger,args)
    logger.write('Length of training set: %d, length of testing set: %d' %
                 (len(train_set),len(test_set)))
    logger.write('Batch size: %d, Gradient accumulation: %d' %
                 (args.batch_size,gradient_accumulation_steps))
    logger.write('Number of device: %d' % (torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    #initialization of trainer from huggingface
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=test_set,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_step if args.val_set_size > 0 else None,
            save_steps=args.save_step,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end= False,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to=None,
            run_name=None
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    rewrite_train(trainer,tokenizer,args,logger)