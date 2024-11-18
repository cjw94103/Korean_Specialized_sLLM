from vllm import LLM, SamplingParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from utils import *

import numpy as np
import pandas as pd
import argparse

# for solar instruct
def make_chat_template(system_contents, user_contents, assistant_contents=None, add_generation_prompt=False):
    if add_generation_prompt == False:
        chat_template = '<s>' + "### System:" + "\n" + system_contents + "\n\n### User:\n" + user_contents + "\n\n### Assistant:\n" + assistant_contents + '</s>'
    elif add_generation_prompt == True:
        chat_template = '<s>' + "### System:" + "\n" + system_contents + "\n\n### User:\n" + user_contents + "\n\n### Assistant:\n"

    return chat_template

## argparse
parser = argparse.ArgumentParser()

# prepare dataset
parser.add_argument("--testset_path", type=str, help="your testset path", default="./sample_data/TaskDataset_sample.jsonl")

# prepare trained LLM
parser.add_argument("--model_name", type=str, help="your Trained model name, supported llama3.1, solar", default="llama3.1")
parser.add_argument("--model_id", type=str, help="your Trained model path", default="/data/GenericLLM_weights/Generic_LLaMA_3.1_8B_DPO/")
parser.add_argument("--tensor_parallel_size", type=int, help="Number of GPUs to use for inference", default=1)
parser.add_argument("--gpu_memory_utilization", type=float, help="Percentage of model memory to allocate to the GPU", default=0.8)

# decoding parameter (sampling parameter)
parser.add_argument("--max_tokens", type=int, help="Maximum number of tokens to generate", default=4096)
parser.add_argument("--temperature", type=float, help="Temperature parameter", default=0.3)
parser.add_argument("--presence_penalty", type=float, default=0.0)
parser.add_argument("--frequency_penalty", type=float, default=0.0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=-1)
parser.add_argument("--min_p", type=float, default=0.0)
parser.add_argument("--use_beam_search", action="store_true")
parser.add_argument("--length_penalty", type=float, default=1.0)

# output save path
parser.add_argument("--output_path", type=str, help="inference output path", default="./output.jsonl")

args = parser.parse_args()

# load tokenizer
if args.model_name == "llama3.1":
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# load llm
llm = LLM(model=args.model_id, 
          tensor_parallel_size=args.tensor_parallel_size, 
          gpu_memory_utilization=args.gpu_memory_utilization)

# decoding parameter
sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, presence_penalty=args.presence_penalty,
                                frequency_penalty=args.frequency_penalty, repetition_penalty=args.repetition_penalty, top_p=args.top_p,
                                top_k=args.top_k, min_p=args.min_p, use_beam_search=args.use_beam_search, length_penalty=args.length_penalty)

# data preprocessing
testset = load_jsonl_file(args.testset_path)
input_text_list = []

for data in testset:
    system_contents, user_contents, assistant_contents = data['system_prompt'], data['user_content'], data['assistant_content']
    if args.model_name == 'llama3.1':
        input_text = tokenizer.apply_chat_template([{'role': 'system', 'content' : system_contents}, {'role': 'user', 'content' : user_contents}],
                                                   tokenize=False, add_generation_prompt=True)
    elif args.model_name == 'solar':
        input_text = make_chat_template(system_contents, user_contents, add_generation_prompt=True)
        
    input_text_list.append(input_text)

# inference
outputs = llm.generate(input_text_list, sampling_params)

output_text_list = []
for output in outputs:
    output_text_list.append(output.outputs[0].text)

# output save jsonl
save_jsonl(args.output_path, output_text_list)