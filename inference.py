import torch
import transformers
import glob
import numpy as np
import argparse

from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer
from utils import *

## argparse
parser = argparse.ArgumentParser()

# prepare dataset
parser.add_argument("--validation_path", type=str, help="your validation set path", default="../00_Data/02_TrainValidTest/02_02_Valset_SFT_llama31.jsonl")

# prepare trained LLM
parser.add_argument("--max_seq_length", type=int, help="Choose according to your training data", default=8196)
parser.add_argument("--dtype", type=int, help="dtype of None is auto detecting", default=None)
parser.add_argument("--load_in_4bit", action='store_false', help="Whether to load in 4 bits, True recommended")
parser.add_argument("--model_id", type=str, help="pretrained model id or local path", default="Enkeeper/LLaMA3.1_TaskInstruct_LoRA_DPO")
parser.add_argument("--model_name", type=str, help="pretrained model name for chat template, llama31, solar etc", default="llama31")
parser.add_argument("--save_16bit_path", type=str, help="save merged 16 bit weights for vLLM", default="./weights")

## 필요 함수
def make_chat_dict(system_contents, user_contents):
    return [{'role':'system', 'content':system_contents}, {'role':'user', 'content':user_contents}]

## load trained model
max_seq_length = args.max_seq_length
dtype = args.dtype
load_in_4bit = args.load_in_4bit

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_id, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit)

FastLanguageModel.for_inference(model)

## make Streamer instance
streamer = TextStreamer(tokenizer, skip_prompt=True)

## Preprocess input text
system_content = """질문에 대한 올바른 답변을 상세하고 면밀하게 작성해 주세요. 아래와 같은 규칙을 지켜주세요.
1. 답변은 심층적 분석을 포함하며, 전문적인 어조를 사용해야합니다.
2. 답변은 질문에 대한 깊이 있는 설명과 예시를 제공해야 합니다.
3. 답변은 bullet points를 사용해 독자가 쉽게 이해할 수 있는 명확한 계층 구조를 갖춰야 합니다. 
4. 모든 답변은 Markdown format으로 작성되어야 하며, 명시적으로 ```markdown ```을 사용하지 않도록 주의해서 작성하세요."""
user_content = '###Question###:\n양자 컴퓨팅이란 무엇인가?\n\n###Answer###:'

input_message = make_chat_dict(system_content, user_content)
input_message = tokenizer.apply_chat_template(input_message, tokenize=False, add_generation_prompt=True)
input_tokens = tokenizer(input_message, return_tensors='pt', return_token_type_ids=False)

## Grid Search Decoding
result = model.generate(**input_tokens,
                        max_new_tokens=1024, early_stopping=True, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, 
                        temperature=0.3, streamer=streamer)

## Save Merged 16 bis for VLLM
model.save_pretrained_merged(args.save_16bit_path, tokenizer, save_method="merged_16bit")
