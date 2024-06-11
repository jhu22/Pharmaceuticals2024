import smiles_llm as llm
filename = "data/dpo_data.json"
checkpoint = "checkpoints/dpo10kUV"

hyperparams = {"batch_size": 256, "max_epochs": 30, "min_epochs": 15,
               "max_length": 64, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 8, "n_embd": 12 * 48}


num_workers = 4  # Number of dataloader worker processes.

tokenizer = llm.SMILESBPETokenizer(dropout=None)

tokenizer = llm.SMILESBPETokenizer.get_hf_tokenizer('checkpoints/10k/tokenizer.json', model_max_length=hyperparams["max_length"])

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
config = GPTNeoXConfig(vocab_size = tokenizer.vocab_size,
                       hidden_size=256,
                       intermediate_size = 1024,
                       max_position_embeddings = 128,
                       num_hidden_layer = 12,
                       num_attention_heads = 16,
                       hidden_act = 'gelu',
                       rotary_pct = 0.25,
                       rotary_emb_base = 10000,
                       attention_dropout = 0.0,
                       hidden_dropout = 0.0,
                       initializer_range = 0.02,
                       layer_norm_eps = 1e-05,
                       use_cache = True,
                       bos_token_id = tokenizer.bos_token_id,
                       eos_token_id = tokenizer.eos_token_id,
                       tie_word_embeddings = False,
                       use_parallel_residual = True,
                       rope_scaling = None,
                       ttention_bias = True,
                       )
model = GPTNeoXForCausalLM(config)
model = GPTNeoXForCausalLM.from_pretrained("checkpoints/10kUV/model", output_attentions=True)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

import json
def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

data_dict = load_json_file('data/dpo_data.json')

from datasets import Dataset
dataset = Dataset.from_dict(data_dict)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",  # 训练结果的输出目录
    overwrite_output_dir=True,  # 是否覆盖之前的输出目录
    num_train_epochs=15,  # 训练的 epochs 数
    per_device_train_batch_size=8,  # 每个设备的训练批量大小
    save_steps=100,  # 每隔多少步保存一次模型
    logging_steps=100,  # 每隔多少步记录一次日志信息
    save_total_limit=10,  # 最多保存多少个模型
    evaluation_strategy="steps",  # 在何时进行评估（steps 或 epoch）
    eval_steps=100,  # 每隔多少步进行一次评估
    logging_dir="./logs",  # 日志输出目录
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
    # 更多参数可以根据需要设置
)

from trl import DPOTrainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    beta=0.1,
    args = training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)
dpo_trainer.train()
model.save_pretrained("checkpoints/dpo10kUV/model/")

import tqdm
model.eval()

import torch
generated_smiles_list = []
n_generated = 30_00
for _ in tqdm.tqdm(range(n_generated)):
    
    # Generate from "<s>" so that the next token is arbitrary.
    smiles_start = torch.LongTensor([[tokenizer.bos_token_id]])
    # Get generated token IDs.
    smiles_start = smiles_start.to('cuda')
    generated_ids = model.generate(smiles_start,
                                   max_length=hyperparams["max_length"],
                                   do_sample=True, top_p=hyperparams["top_p"],
                                   pad_token_id=tokenizer.eos_token_id)
    # Decode the IDs into tokens and remove "<s>" and "</s>".
    generated_ids = generated_ids.to('cuda')
    generated_smiles = tokenizer.decode(generated_ids[0],
                                        skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)

import numpy as np
np.save('dponexo-uv-eval.npy', generated_smiles_list)