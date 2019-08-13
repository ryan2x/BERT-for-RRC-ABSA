# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel, BertPreTrainingHeads

from pytorch_pretrained_bert.optimization import BertAdam

import modelconfig
from absa_models import BertForMTPostTraining


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def train(args):
    #load squad data for pre-training.
    
    args.train_batch_size=int(args.train_batch_size / args.gradient_accumulation_steps)
    
    review_train_examples=np.load(os.path.join(args.review_data_dir, "data.npz") )
    squad_train_examples=np.load(os.path.join(args.squad_data_dir, "data.npz") )
    
    num_train_steps = args.num_train_steps
    
    # load bert pre-train data.
    review_train_data = TensorDataset(
        torch.from_numpy(review_train_examples["input_ids"]),
        torch.from_numpy(review_train_examples["segment_ids"]),
        torch.from_numpy(review_train_examples["input_mask"]),           
        torch.from_numpy(review_train_examples["masked_lm_ids"]),
        torch.from_numpy(review_train_examples["next_sentence_labels"]) )
    
    review_train_dataloader = DataLoader(review_train_data, sampler=RandomSampler(review_train_data), batch_size=args.train_batch_size , drop_last=True)
    
    squad_train_data = TensorDataset(
        torch.from_numpy(squad_train_examples["input_ids"]), 
        torch.from_numpy(squad_train_examples["segment_ids"]), 
        torch.from_numpy(squad_train_examples["input_mask"]), 
        torch.from_numpy(squad_train_examples["start_positions"]), 
        torch.from_numpy(squad_train_examples["end_positions"] ) )
    
    squad_train_dataloader = DataLoader(squad_train_data, sampler=RandomSampler(squad_train_data), batch_size=args.train_batch_size , drop_last=True)

    #we do not have any valiation for pretuning
    model = BertForMTPostTraining.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model] )
    
    if args.fp16:
        model.half()
    model.cuda()
        
    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
        
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)

        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step=0
    step=0
    batch_loss=0.
    model.train()
    model.zero_grad()
    
    training=True
    
    review_iter=iter(review_train_dataloader)
    squad_iter=iter(squad_train_dataloader)
    
    while training:
        try:
            batch = next(review_iter)
        except:
            review_iter=iter(review_train_dataloader)
            batch = next(review_iter)
            
        batch = tuple(t.cuda() for t in batch)
        
        input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_labels = batch
        
        review_loss = model("review", input_ids.long(), segment_ids.long(), input_mask.long(), masked_lm_ids.long(), next_sentence_labels.long(), None, None)
        
        try:
            batch = next(squad_iter)
        except:
            squad_iter=iter(squad_train_dataloader)
            batch = next(squad_iter)

        batch = tuple(t.cuda() for t in batch)
        input_ids, segment_ids, input_mask, start_positions, end_positions = batch
        
        squad_loss = model("squad", input_ids.long(), segment_ids.long(), input_mask.long(), None, None, start_positions.long(), end_positions.long() )

        loss=review_loss + squad_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        batch_loss+=loss
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 50 ==0:
                logging.info("step %d batch_loss %f ", global_step, batch_loss)
            batch_loss=0.

            if global_step % args.save_checkpoints_steps==0:
                model.float()
                torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model_"+str(global_step)+".bin") )
                if args.fp16:
                    model.half()
            if global_step>=num_train_steps:
                training=False
                break
        step+=1
    model.float()
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin") )

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base", type=str, required=True, help="pretrained weights of bert.")

    parser.add_argument("--review_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="dir of review numpy file dir.")
    
    parser.add_argument("--squad_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="dir of squad preprocessed numpy file.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--train_batch_size", default=16,
                        type=int, help="training batch size for both review and squad.")
        
    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")

    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_steps", default=50000,
                        type=int, help="Number of training steps.")

    parser.add_argument("--warmup_proportion", default=0.1,
                        type=float, help="Number of warmup steps.")

    parser.add_argument("--save_checkpoints_steps", default=5000,
                        type=int, help="How often to save the model checkpoint.")

    parser.add_argument("--seed", default=12345,
                        type=int, help="random seed.")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2)
    
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)        
        train(args)

if __name__ == "__main__":
    main()
