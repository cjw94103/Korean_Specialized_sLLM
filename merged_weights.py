import torch
import transformers
import glob
import numpy as np
import argparse

from unsloth import FastLanguageModel

# argparse
# prepare trained LLM
parser.add_argument("--model_id", type=str, help="Trained SFT model path", default="model_result/02_01_SoLAR_10.7B_SFT/checkpoint-2757")
parser.add_argument("--max_seq_length", type=int, help="Choose according to your training data", default=8196)
parser.add_argument("--dtype", type=int, help="dtype of None is auto detecting", default=None)
parser.add_argument("--load_in_4bit", action='store_false', help="Whether to load in 4 bits, True recommended")

# save merged weights path
parser.add_argument("--merged_weights_path", type=str, help="merged 16 bit weights save path", default="/data/GenericLLM_weights/Generic_SoLAR_DPO/")

args = parser.parse_args()

# load trained LLM
max_seq_length = args.max_seq_length
dtype = args.dtype
load_in_4bit = args.load_in_4bit

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit)
FastLanguageModel.for_inference(model)

# save weights merged 16bit format
model.save_pretrained_merged(args.merged_weights_path, tokenizer, save_method="merged_16bit")