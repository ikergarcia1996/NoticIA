<p align="center">
    <img src="assets/logo.png" style="height: 250px;">
</p>

<h3 align="center">"A Spanish dataset for Clickbait articles summarization"</h3>

<p align="center">
    <a href="https://twitter.com/intent/tweet?text=NoticIA+el+dataset+de+resumen+de+noticias+clickbait+en+español:&url=https%3A%2F%2Fgithub.com%2Fikergarcia1996%2FNoticIA"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fikergarcia1996%2FNoticIA"></a>
    <a href="https://github.com/ikergarcia1996/NoticIA/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/ikergarcia1996/NoticIA"></a>
<a href="https://huggingface.co/datasets/Iker/NoticIA"><img alt="Public Dataset" src="https://img.shields.io/badge/🤗HuggingFace-Dataset-green"></a>
    <a href="https://visitor-badge.laobi.icu/badge?page_id=ikergarcia1996.noticia"><img src="https://visitor-badge.laobi.icu/badge?page_id=ikergarcia1996.noticia" alt="visitor badge"/></a>
    <a href="https://arxiv.org/abs/2404.07611"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
     <br>
     <br>
</p>


We present NoticIA, a dataset consisting of 850 Spanish news articles featuring prominent clickbait headlines, each paired with high-quality, single-sentence generative summarizations written by humans.


- 📖 Paper: [NoticIA: A Clickbait Article Summarization Dataset in Spanish](https://arxiv.org/abs/2404.07611)
- 💻 Dataset: [https://hf.co/datasets/Iker/NoticIA](https://huggingface.co/datasets/Iker/NoticIA)
- 💻 Dataset (Instruction format): [https://hf.co/datasets/somosnlp/NoticIA-it](https://huggingface.co/datasets/somosnlp/NoticIA-it)
- 🤖 Pre Trained Models [https://hf.co/collections/Iker/noticia-and-clickbaitfighter-65fdb2f80c34d7c063d3e48e](https://huggingface.co/collections/Iker/noticia-and-clickbaitfighter-65fdb2f80c34d7c063d3e48e)
- 🔌 Online Demo: [https://hf.co/spaces/somosnlp/NoticIA-demo](https://huggingface.co/spaces/somosnlp/NoticIA-demo)


For example, given the following headline and web text:
```
# ¿Qué pasará el 15 de enero de 2024?
Al parecer, no todo es dulzura en las vacaciones de fin de años, como lo demuestra la nueva intrig....
```
The summary is:
```
Que los estudiantes vuelven a clase.
```

<table>
<tr>   
<td style="width:100%"><img src="https://github.com/ikergarcia1996/NoticIA/raw/main/results/Results.png" align="right" width="100%"> </td>
</tr>
</table>


## Dataset

The easiest and recommended way to download the dataset is using the 🤗HuggingFace Hub. See the [Dataset Card](https://huggingface.co/datasets/Iker/NoticIA) for more information about the dataset.

```Python
from datasets import load_dataset
dataset = load_dataset("Iker/NoticIA")
```

We also distribute the dataset in '.jsonl' format in this repository as a backup. See [dataset/README.md](dataset/) for more details.

# How to evaluate an LLM in noticIA

We provide a script to evaluate any LLM in the dataset. First, you need to create a configuration file. See [configs/sample_configs](configs/sample_configs) and [configs/configs_zero-shot](configs/configs_zero-shot) for examples.
The following config will evaluate the `NousResearch/Nous-Hermes-2-SOLAR-10.7B` model in our dataset in zero-shot setting.

```yaml
#Model name in the HF Hub or path to a local directory that contains the model
model_name_or_path: NousResearch/Nous-Hermes-2-SOLAR-10.7B
# dtype in which we will load the model. You should use bfloat unless you hardware doesn't support it. In that case use float16
torch_dtype: bfloat16
# Load models that require custom code in the HF Hub (warning: This setting will run arbitrary code on the HUB)
trust_remote_code: true
# Performs quatization using bitsandbytes integration. Allows evaluating LLMs in consumer hardware (4 for 4-bit quantization, 8 for 8-bit quantization, null for no quantization)
quantization_inference: null
#If force_auto_device_map is set to True. We will split the model into all the available GPUs and CPU, this is useful for large models that do not fit in a single GPU VRAM. 
force_auto_device_map: false
# Use Flash Attention for Fast and memory-efficient attention.
use_flash_attention: true
# Text generation hyperparameters. By default we use gready-search
generation_config: generation_config.json
# Stop words that will end the generation. Each model has different stop words depending on the Chat template it uses. See configs/configs_zero-shot for examples. 
stop_words:
  - "<|im_end|>"
  - "<|im_start|>"
  - "<|im_start|>system"
  - "<|im_start|>user"
  - "<|im_start|>assistant"
  - "</s>"
  - "<s>"
  - "\\n"

# Name of the split (we will download it from the HF HUB) or path to a local .jsonl file that contains the dataset
# You can add multiple dataset paths. You can use the "all" split to evaluate the model on the whole dataset.
test_datasets: 
  - test
# Max sequence length (prompt+answer)
max_seq_length: 8192
# Max tokens that we will generate (answer)
generation_max_length: 8192

# Output dir where we will sabe the predictions
output_dir: results/zero/NousResearch_Nous-Hermes-2-SOLAR-10.7B
# Overwrite the output dir if it exists. 
overwrite_output_dir: true

# Run training
do_train: false
# Run evaluation in the development set during training
do_eval: false
# Run inference
do_predict: true
# If set to false, we will sample the probability of generating the gold summary (compute loss). If set to true, the model will generate a text and we will evaluate it using the rouge score between the gold and the predictions.
predict_with_generate: true
# Batch size for evaluation.
per_device_eval_batch_size: 4


# Settings to report the results to wandb. 
logging_strategy: steps
logging_first_step: true
logging_steps: 5
# Set to none if you don't want to report the results to wandb.
report_to: wandb
run_name: "Nous-Hermes-2-SOLAR-10.7B"
disable_tqdm: false

# dtype in which we will run the compatition. You should use bfloat unless you hardware doesn't support it. In that case use float16
bf16: true
fp16: false
```

Once you have create the config file, you can run the evaluation script:

```bash
python3 run.py path_to_your_config/Nous-Hermes-2-SOLAR-10.7B.yaml
```

You can use accelerate to run the evaluation in multiple GPUs. 
```bash
accelerate launch --multi_gpu --num_processes 2 run.py path_to_your_config/Nous-Hermes-2-SOLAR-10.7B.yaml
```

### Running Large Models that do not fit into GPU VRAM

Large models do not fit in a single GPU. If you run into out-of-memory issues you can follow two approaches

#### Model quantization
By using the Huggingface trainer integration of [bits-and-bytes](https://github.com/TimDettmers/bitsandbytes) we can 
can perform 4-bit quantization of the model weights. A 4-bit quantized model will require ~4 times less memory than 
running the model in 16 bits. In order to quantize the model, we will modify the previous `config.yaml` file as follows:
```yaml
# Performs quatization using bitsandbytes integration. Allows evaluating LLMs in consumer hardware (4 for 4-bit quantization, 8 for 8-bit quantization, null for no quantization)
quantization_inference: 4
#If force_auto_device_map is set to True. We will split the model into all the available GPUs and CPU, this is useful for large models that do not fit in a single GPU VRAM. 
force_auto_device_map: true
```

We also set `force_auto_device_map` to `true`. This will offload some of the model parameters to the CPU if they don't feed into
the GPU VRAM. In case that you have multiple GPUs it will first split the model across GPUs. 

#### Deepspeed

If you have multiple GPUs available, you can use deepspeed zero 3 to shard the model parameters across the GPUs.
This will efficiently split the model across GPUs using tensor parallelism. This allows to perform fast inference
using very large language models. In order to use deepspeed you just need to provide a deepspeed configuration 
into the `config.yaml` file. We already provide the [configs/deepspeed_configs/deepspeed_zero3.json](configs/deepspeed_configs/deepspeed_zero3.json) config file.

```yaml
# Deepspeed requires the model to be loaded either in bfloat16 or float16. 
torch_dtype: bfloat16
# Currently deepspeed is not compatible with  4-bit quantization or 8-bit quantization.
quantization_inference: null
# Required for deepspeed
force_auto_device_map: false
# Path to the deepspeed configuration
deepspeed: configs/deepspeed_configs/deepspeed_zero3.json
```

You can run the inference using deepspeed with the following command. This will shard the model parameters across 4 GPUs.
```bash
accelerate launch --multi_gpu --num_processes 4 run.py path_to_your_config/Nous-Hermes-2-SOLAR-10.7B_Deepspeed.yaml
```

# Change the default prompt

The file [prompts.py](prompts.py) contains the prompts that we use in our experiments. By 
default we use the `summarize_clickbait_short_prompt` prompt. You can add any prompt you want
to this file. To define the prompt to use as follows. This config will evaluate the test set 
3 times, each one using a different prompt. It will save the results into separate files with the
name `{dataset_name}_{prompt_name}`.
```yaml
test_datasets: 
  - test
  - test
  - test
test_datasets_prompts:
  - summarize_clickbait_short_prompt
  - summarize_clickbait_large_prompt
  - your_custom_prompt
```

# Training a model with NoticIA
You can train a LLMs in our dataset. First, you need to create a configuration file. See [configs/configs_finetune](configs/configs_finetune) for examples. Here is an example config to finetune `NousResearch/Nous-Hermes-2-SOLAR-10.7B`.
You can follow two different approaches. If have multiple GPUs you can train all the model parameters using Deepspeed to 
shard the model parameters, gradients and optimizers across GPUs. If you have a single GPU or not enough GPU memory, you can 
combine 4-bit quantization and [LoRA](https://huggingface.co/docs/diffusers/training/lora) to efficiently finetune models on a single GPU. 

### Example using Deepspeed
```bash
#Training args
model_name_or_path: NousResearch/Nous-Hermes-2-SOLAR-10.7B
torch_dtype: bfloat16
use_lora: false
quantization: null
quantization_inference: null
gradient_checkpointing: true
force_auto_device_map: false
use_flash_attention: true
deepspeed: configs/deepspeed_configs/deepspeed_zero3.json
generation_config: generation_config.json
stop_words:
  - "<s>"
  - "</s>"
  - "\\n"
  - "### User:"
  - "### Assistant:"
  - "###"

# dataset arguments
train_datasets: 
  - train
validation_datasets: 
  - validation
test_datasets: 
  - test
max_seq_length: 8192
generation_max_length: 8192
prompt_loss_weight: 0.0

# checkpoint settings
output_dir: ../models_all/Nous-Hermes-2-SOLAR-10.7B_bs64_lr_0.000005_epoch3
overwrite_output_dir: true
load_best_model_at_end: true
metric_for_best_model: eval_test_test/rouge
greater_is_better: true
save_strategy: "epoch"
save_only_model: true
save_total_limit: 1

# evaluation
do_train: true
do_eval: true
do_predict: true
evaluation_strategy: "epoch"
predict_with_generate: true
evaluate_all_checkpoints: true

# batch size: 4 batch size * 4 gradaccum * 4 GPUs = 64
per_device_train_batch_size: 4
per_device_eval_batch_size: 2
gradient_accumulation_steps: 4
generation_num_beams: 1

# optimizer settings
optim: adamw_torch_fused
learning_rate: 0.000005
weight_decay: 0.0
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1e-12

# reporting
logging_strategy: steps
logging_first_step: true
logging_steps: 5
report_to: wandb
run_name: "Nous-Hermes-2-SOLAR-10.7B_bs64_lr_0.000005_epoch3"
disable_tqdm: false

# hub settings
push_to_hub: false
resume_from_checkpoint: false

# performance
bf16: true
fp16: false
torch_compile: false
ddp_find_unused_parameters: false
```
Run the training with the following command
```bash
accelerate launch --multi_gpu --num_processes 4 run.py path_to_your_config/Nous-Hermes-2-SOLAR-10.7B_Deepspeed.yaml
```

### Example using 4-bit Quantization + LoRA
```bash
#Training args
model_name_or_path: NousResearch/Nous-Hermes-2-SOLAR-10.7B
torch_dtype: bfloat16
use_lora: false
quantization: null
quantization_inference: null
gradient_checkpointing: true
force_auto_device_map: false
use_flash_attention: true
deepspeed: configs/deepspeed_configs/deepspeed_zero3.json
generation_config: generation_config.json
stop_words:
  - "<s>"
  - "</s>"
  - "\n"
  - "### User:"
  - "### Assistant:"
  - "###"

# dataset arguments
train_datasets: 
  - train
validation_datasets: 
  - validation
test_datasets: 
  - test
max_seq_length: 8192
generation_max_length: 8192
prompt_loss_weight: 0.0

# checkpoint settings
output_dir: ../models_all/Nous-Hermes-2-SOLAR-10.7B_Lora
overwrite_output_dir: true
load_best_model_at_end: true
metric_for_best_model: eval_test_test/rouge
greater_is_better: true
save_strategy: "epoch"
save_only_model: true
save_total_limit: 1

# evaluation
do_train: true
do_eval: true
do_predict: true
evaluation_strategy: "epoch"
predict_with_generate: true
evaluate_all_checkpoints: true

# batch size: 8 batch size * 8 gradaccum * 1 GPUs = 64
per_device_train_batch_size: 8
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8
generation_num_beams: 1

# optimizer settings
optim: adamw_torch_fused
learning_rate: 0.0003
weight_decay: 0.0
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1e-12

# lora settings
lora_r: 128
lora_alpha: 256
lora_dropout: 0.05
lora_target_modules:
  - all

# reporting
logging_strategy: steps
logging_first_step: true
logging_steps: 5
report_to: wandb
run_name: "Nous-Hermes-2-SOLAR-10.7B_Lora"
disable_tqdm: false

# hub settings
push_to_hub: false
resume_from_checkpoint: false

# performance
bf16: true
fp16: false
torch_compile: false
ddp_find_unused_parameters: false
```

Run the training using the following command:
```bash
python3 run.py path_to_your_config/Nous-Hermes-2-SOLAR-10.7B.yaml
```
# Citation

```bittext
@misc{noticia2024,
      title={NoticIA: A Clickbait Article Summarization Dataset in Spanish}, 
      author={Iker García-Ferrero and Begoña Altuna},
      year={2024},
      eprint={2404.07611},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
