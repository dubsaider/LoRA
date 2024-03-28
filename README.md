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

## Woman cartoon

### Base SDXL

![image (10)](https://github.com/dubsaider/LoRA/assets/44964427/627e1546-2e90-49c4-9b3d-33208dac81a3)

### SDXL LoRA

![image (11)](https://github.com/dubsaider/LoRA/assets/44964427/2bf44d26-ad3e-4181-9f7f-bfe4dfaa8c83)

## Cat cartoon

### Base SDXL

![image (13)](https://github.com/dubsaider/LoRA/assets/44964427/af0cbf18-9540-4c43-9d9e-7c61b869fcce)

### SDXL LoRA

![image (12)](https://github.com/dubsaider/LoRA/assets/44964427/9940416c-6702-47ff-a22f-020a07be0c6f)

