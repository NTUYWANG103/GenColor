# GenColor: Generative and Expressive Color Enhancement with Pixel-Perfect Texture Preservation
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yidong.pro/projects/gencolor/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=n8AvXKcCeR)


Official PyTorch Implementation of NeurIPS 2025 (Spotlight) Paper "[GenColor: Generative and Expressive Color Enhancement with Pixel-Perfect Texture Preservation](https://openreview.net/pdf?id=n8AvXKcCeR)"



![Teaser](asset/Teaser.png)
![Pipeline](asset/Pipeline.png)

# Environment
```bash
conda create -n gencolor python=3.10 -y
conda activate gencolor
pip install -r requirements.txt
```

# Inference

###  Model Checkpoint
Please download the model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1YyJGZdA9TUChw8gLLR-Dv127Z9j4iFNg?usp=sharing), then run the demo script. 

Please note that due to commercialization and copyright considerations, only stage 2 texture preservation network is released. Please stay tuned for the release of the stage 1 color enhancement model.

### Single image enhancement demo script
```bash
python demo.py --input example/tree.png --output example/tree_result.png
```

### Fusion demo notebook

Run the fusion demo notebook to combine the stage 1 color enhancement with the stage 2 texture preservation network:


```bash
demo_fusion.ipynb
```




# Training

### ARTISAN-1.2M Dataset
Before training, you need to download the [ARTISAN-1.2M dataset](http://47.254.153.212:8501/) and generate captions for your images. 

**Note**: Due to commercialization and copyright considerations, we release a 240P preview version of the dataset.

We support two models for caption generation: **BLIP** and **Qwen3-VL**.

### Using BLIP
```bash
python training/stage1/image_caption.py \
--image_dir="data/ARTISAN-1.2M/images" \
--output_json="data/ARTISAN-1.2M/captions.json" \
--model_type blip \
--batch_size 8
```

### Using Qwen3-VL
```bash
python training/stage1/image_caption.py \
--image_dir="data/ARTISAN-1.2M/images" \
--output_json="data/ARTISAN-1.2M/captions.json" \
--model_type qwen \
--batch_size 1 \
--device_map auto
```

### Then train the stage1 model:

```bash
CUDA_VISIBLE_DEVICES=0 python training/stage1/train_controlnet.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--img_dir="data/ARTISAN-1.2M/images" \
--prompt_json_path="data/ARTISAN-1.2M/captions.json" \
--output_dir="ckpt/GenColor_SD21" \
--resolution=512 \
--learning_rate=1e-5 \
--train_batch_size=16 \
--num_train_epochs=100000 \
--tracker_project_name="GenColor_SD21-stage1" \
--checkpointing_steps=10000 \
--report_to wandb \
--validation_steps=500 \
--lr_warmup_steps=0 \
--mixed_precision="bf16" \
--dataloader_num_workers=8 \
--resume_from_checkpoint='latest'
```

# Acknowledgments
This code repository is partially borrowed from [Harmonizer](https://github.com/ZHKKKe/Harmonizer) and [SwinIR](https://github.com/JingyunLiang/SwinIR).

# Citation
You can cite it as follows:

```
@inproceedings{donggencolor,
  title={GenColor: Generative and Expressive Color Enhancement with Pixel-Perfect Texture Preservation},
  author={Dong, Yi and Wang, Yuxi and Lin, Xianhui and Ouyang, Wenqi and Shen, Zhiqi and Ren, Peiran and Fan, Ruoxi and Lau, Rynson WH},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```