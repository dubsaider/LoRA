## LoRA

# Model
<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>

# Dataset
<https://huggingface.co/datasets/Norod78/cartoon-blip-captions>

# Install 
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements_sdxl.txt
```

# Use
```
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="Norod78/cartoon-blip-captions"

accelerate launch train_text_to_image_lora_sdxl.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_NAME \
 --caption_column="text" \
 --dataloader_num_workers=8 \
 --resolution=512 \
 --random_flip \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --max_train_steps=15000 \
 --learning_rate=1e-04 \
 --max_grad_norm=1 \
 --lr_scheduler="cosine" \
 --lr_warmup_steps=0 \
 --report_to=wandb \
 --checkpointing_steps=500 \
 --validation_prompt="A cartoon character with blue eyes." \
 --seed=1337
```
