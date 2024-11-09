import torch
import transformers
import argparse

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer

## argparse
parser = argparse.ArgumentParser()

# prepare dataset
parser.add_argument("--trainset_path", type=str, help="your train set path", default="../00_Data/02_TrainValidTest/01_02_Trainset_SFT_llama31.jsonl")
parser.add_argument("--validation_path", type=str, help="your validation set path", default="../00_Data/02_TrainValidTest/02_02_Valset_SFT_llama31.jsonl")

# prepare pretrained LLM
parser.add_argument("--max_seq_length", type=int, help="Choose according to your training data", default=8196)
parser.add_argument("--dtype", type=int, help="dtype of None is auto detecting", default=None)
parser.add_argument("--load_in_4bit", action='store_false', help="Whether to load in 4 bits, True recommended")
parser.add_argument("--model_id", type=str, help="pretrained model id or local path", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

# LoRA Parameter
parser.add_argument("--lora_r", type=int, help="Choose LoRA r parameter", default=64)
parser.add_argument("--lora_alpha", type=int, help="Choose LoRA alpha parameter", default=128)
parser.add_argument("--lora_dropout", type=float, help="Choose LoRA dropout parameter", default=0.)
parser.add_argument("--use_rslora", action='store_true', help="Whether to use rslora, False recommended")

# training argument
parser.add_argument("--train_epochs", type=int, help="num epochs", default=5)
parser.add_argument("--batch_size", type=int, help="per_device_batch_size", default=4)
parser.add_argument("--gradient_accum_steps", type=int, help="gradient_accumulation_steps", default=8)
parser.add_argument("--weight_decay", type=float, help="l2 weight decay", default=0.1)
parser.add_argument("--save_steps", type=int, help="choose save steps", default=919)
parser.add_argument("--learning_rate", type=float, help="choose learning rate", default=2e-5)
parser.add_argument("--output_dir", type=str, help="trained model save path", default="model_result/01_01_llama3.1_8B_SFT")
parser.add_argument("--save_total_limit", type=int, help="How many models will be saved during training?", default=5)

args = parser.parse_args()

## load dataset
data_files = {"train": args.trainset_path, "validation": args.validation_path}
data = load_dataset('json', data_files=data_files)

## Load Pretrained sLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_id,
    max_seq_length = args.max_seq_length,
    dtype = args.dtype,
    load_in_4bit = args.load_in_4bit)

## get peft model
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# (r, alpha) = [(8, 16), (16, 32), (32, 64), (64, 128), (128, 256)??] 일단 (64, 128)로 dpo 단계까지 해보기!
model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = args.use_rslora,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print_trainable_parameters(model)

# get training argument
training_args = TrainingArguments(
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        learning_rate=args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=args.save_steps,
        logging_strategy="steps",
        output_dir=args.output_dir,
        optim="adamw_8bit",
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit
    )

# get trainer
trainer = SFTTrainer(model=model, 
                     tokenizer = tokenizer,
                     args=training_args, 
                     train_dataset=data["train"], 
                     eval_dataset=data['validation'],
                     dataset_text_field = "text",
                     max_seq_length = args.max_seq_length,
                     packing = False)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train!!
trainer_stats = trainer.train()