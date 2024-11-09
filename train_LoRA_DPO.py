import torch
import transformers
import argparse

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments
from unsloth import PatchDPOTrainer

# dpo mode
PatchDPOTrainer()

## argparse
parser = argparse.ArgumentParser()

# prepare dataset
parser.add_argument("--trainset_path", type=str, help="your train set path", default="../00_Data/02_TrainValidTest/01_03_Trainset_DPO_llama31.jsonl")
parser.add_argument("--validation_path", type=str, help="your validation set path", default="../00_Data/02_TrainValidTest/02_03_Valset_DPO_llama31.jsonl")

# prepare pretrained LLM
parser.add_argument("--max_seq_length", type=int, help="Choose according to your training data", default=8196)
parser.add_argument("--dtype", type=int, help="dtype of None is auto detecting", default=None)
parser.add_argument("--load_in_4bit", action='store_false', help="Whether to load in 4 bits, True recommended")
parser.add_argument("--model_id", type=str, help="Trained SFT model path", default="model_result/01_01_llama3.1_8B_SFT/checkpoint-4595")

# Dpo Parameter
parser.add_argument("--beta", type=float, help="dpo Beta parameter", default=0.1)

# training argument
parser.add_argument("--train_epochs", type=int, help="num epochs", default=5)
parser.add_argument("--batch_size", type=int, help="per_device_batch_size", default=4)
parser.add_argument("--gradient_accum_steps", type=int, help="gradient_accumulation_steps", default=8)
parser.add_argument("--weight_decay", type=float, help="l2 weight decay", default=0.0)
parser.add_argument("--warmup_ratio", type=float, help="ratio of learning rate warmup ", default=0.1)
parser.add_argument("--save_steps", type=int, help="choose save steps", default=919)
parser.add_argument("--learning_rate", type=float, help="choose learning rate", default=5e-7)
parser.add_argument("--output_dir", type=str, help="trained model save path", default="model_result/01_02_llama3.1_8B_DPO")
parser.add_argument("--save_total_limit", type=int, help="How many models will be saved during training?", default=5)

args = parser.parse_args()

## load dataset
data_files = {"train": args.trainset_path, "validation": args.validation_path}
data = load_dataset('json', data_files=data_files)

## Load SFT Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_id,
    max_seq_length = args.max_seq_length,
    dtype = args.dtype,
    load_in_4bit = args.load_in_4bit)

## get dpo argument
training_args = DPOConfig(
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        learning_rate=args.learning_rate,
        warmup_ratio = args.warmup_ratio,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=args.save_steps,
        logging_strategy="steps",
        output_dir=args.output_dir,
        optim="adamw_8bit",
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit
    )

## get DPO Trainer
trainer = DPOTrainer(model=model, 
                     ref_model = None, # 새로 추가
                     tokenizer = tokenizer,
                     args=training_args, 
                     train_dataset=data["train"], 
                     eval_dataset=data['validation'],
                     beta=args.beta)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

## Train!!
trainer_stats = trainer.train()