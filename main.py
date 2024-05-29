import torch
import GPUtil
import time

from trl import SFTTrainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments

base_model_id = "microsoft/phi-1"

def print_gpu_utilization():
    """Prints GPU usage using GPUtil."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}, Utilization: {gpu.load * 100:.2f}%")

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

# ="flash_attention_2" on Linux
model = AutoModelForCausalLM.from_pretrained(
          base_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map={"": 0}, torch_dtype="auto", attn_implementation="eager"
)


model = prepare_model_for_kbit_training(model)
dataset = load_dataset("timdettmers/openassistant-guanaco")

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","k_proj","v_proj","fc2","fc1"]
)


training_arguments = TrainingArguments(
        output_dir="./phi2-results2",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,
        per_device_eval_batch_size=1,
        log_level="debug",
        save_steps=100,
        logging_steps=25, 
        learning_rate=1e-4,
        eval_steps=50,
        optim='paged_adamw_8bit',
        bf16=True, #change to fp16 if are using an older GPU
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True
)

trainer.train()

duration = 0.0
total_length = 0
prompt = []
#prompt.append("Write the recipe for a chicken curry with coconut milk.")
#prompt.append("Translate into French the following sentence: I love bread and cheese!")
prompt.append("### Human: Cite 20 famous people.### Assistant:")
#prompt.append("Where is the moon right now?")

for i in range(len(prompt)):
  model_inputs = tokenizer(prompt[i], return_tensors="pt").to("cuda:0")
  start_time = time.time()
  output = model.generate(**model_inputs, max_length=500)[0]
  duration += float(time.time() - start_time)
  total_length += len(output)
  tok_sec_prompt = round(len(output)/float(time.time() - start_time),3)
  print("Prompt --- %s tokens/seconds ---" % (tok_sec_prompt))
  print(print_gpu_utilization())
  print(tokenizer.decode(output, skip_special_tokens=True)) 