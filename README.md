# LoRA

## Model
<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>

## Dataset
<https://huggingface.co/datasets/Norod78/cartoon-blip-captions>

### Install 
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements_sdxl.txt
pip install datasets peft wandb huggingface_hub
```

### Use
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

# Examples

## Woman in the simpsons style

### Base SDXL

![00070-1578129095](https://github.com/dubsaider/LoRA/assets/44964427/88f859e4-bc03-406d-b173-af3dd3609a42)

### SDXL LoRA

![00075-3407042086](https://github.com/dubsaider/LoRA/assets/44964427/0ced90e0-a7c6-4551-bb72-d4258382c546)

## Simpsons characters

### Base SDXL

![00055-2428787481](https://github.com/dubsaider/LoRA/assets/44964427/9bb41e91-fe39-4c57-bb02-a7bf489f0619)

### SDXL LoRA

![00065-3135564397](https://github.com/dubsaider/LoRA/assets/44964427/1c6034cc-bfef-471c-bdb2-fedb2de3fe4c)

## Shouth park character

### Base SDXL

![00030-163842441](https://github.com/dubsaider/LoRA/assets/44964427/68d34be4-e253-46ff-af1d-ffefaa057ef4)

### SDXL LoRA

![00035-3447649985](https://github.com/dubsaider/LoRA/assets/44964427/8c552e66-29c4-49cc-8948-74bf7f8324c4)
