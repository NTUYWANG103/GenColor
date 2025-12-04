import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_blip_model(model_path="Salesforce/blip-image-captioning-large", device="cuda"):
    """
    Load BLIP model and processor.
    
    Args:
        model_path (str): Path to the BLIP model or HuggingFace model name.
                         Default: Salesforce/blip-image-captioning-large
        device (str): Device to load the model on. Default: "cuda"
    
    Returns:
        model, processor: Loaded model and processor.
    """
    print(f"Loading BLIP model from {model_path}...")
    print(f"Device: {device}")
    
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print("Model loaded successfully!")
    return model, processor


def load_qwen3vl_model(model_path="Qwen/Qwen3-VL-2B-Instruct", device_map="auto"):
    """
    Load Qwen3VL model and processor with automatic multi-GPU support.
    
    Args:
        model_path (str): Path to the Qwen3VL model or HuggingFace model name.
                         Default: Qwen3-VL-2B-Instruct (lightweight and efficient)
        device_map (str): Device mapping strategy. 
                         "auto" - automatically distribute across all available GPUs (recommended)
                         "balanced" - balanced memory usage across GPUs
                         "sequential" - fill GPUs sequentially
                         Or specific device like "cuda:0" for single GPU
    
    Returns:
        model, processor: Loaded model and processor.
    """
    print(f"Loading Qwen3VL model from {model_path}...")
    print(f"Device map strategy: {device_map}")
    
    # Load model with device_map="auto" for automatic multi-GPU distribution
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,  # This automatically handles multi-GPU
        trust_remote_code=True
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Print device allocation info
    if hasattr(model, 'hf_device_map'):
        print(f"Model distributed across devices: {model.hf_device_map}")
    
    print("Model loaded successfully!")
    return model, processor




def generate_caption_blip(image_path, model, processor, prompt_template=None, max_new_tokens=50):
    """
    Generate a caption for a single image using BLIP.
    
    Args:
        image_path (str): Path to the image file.
        model: BLIP model.
        processor: BLIP processor.
        prompt_template (str): Optional custom prompt template (BLIP uses conditional generation).
        max_new_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        str: Generated caption.
    """
    image = Image.open(image_path).convert('RGB')
    
    if prompt_template:
        # Conditional generation with prompt
        inputs = processor(image, prompt_template, return_tensors="pt").to(model.device)
    else:
        # Unconditional generation
        inputs = processor(image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption


def generate_caption(image_path, model, processor, prompt_template=None, max_new_tokens=50):
    """
    Generate a caption for a single image using Qwen3VL.
    
    Args:
        image_path (str): Path to the image file.
        model: Qwen3VL model.
        processor: Qwen3VL processor.
        prompt_template (str): Optional custom prompt template.
        max_new_tokens (int): Maximum number of tokens to generate.
                             Default is 50 to stay well under SD 2.1's 77 token limit.
    
    Returns:
        str: Generated caption.
    """
    if prompt_template is None:
        prompt_template = "Describe this image in one short sentence."
    
    # Prepare the messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt_template},
            ],
        }
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return caption


def generate_captions_batch_blip(image_paths, model, processor, prompt_template=None, max_new_tokens=50):
    """
    Generate captions for a batch of images using BLIP.
    
    Args:
        image_paths (list): List of image file paths.
        model: BLIP model.
        processor: BLIP processor.
        prompt_template (str): Optional custom prompt template.
        max_new_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        list: List of generated captions.
    """
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    if prompt_template:
        # Conditional generation with prompt
        texts = [prompt_template] * len(images)
        inputs = processor(images, texts, return_tensors="pt", padding=True).to(model.device)
    else:
        # Unconditional generation
        inputs = processor(images, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions


def generate_captions_batch(image_paths, model, processor, prompt_template=None, max_new_tokens=50):
    """
    Generate captions for a batch of images using Qwen3VL.
    
    Args:
        image_paths (list): List of image file paths.
        model: Qwen3VL model.
        processor: Qwen3VL processor.
        prompt_template (str): Optional custom prompt template.
        max_new_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        list: List of generated captions.
    """
    if prompt_template is None:
        prompt_template = "Describe this image in one short sentence."
    
    # Prepare messages for all images
    all_messages = []
    for image_path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_template},
                ],
            }
        ]
        all_messages.append(messages)
    
    # Process batch
    all_texts = []
    all_image_inputs = []
    
    for messages in all_messages:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_texts.append(text)
        image_inputs, _ = process_vision_info(messages)
        all_image_inputs.extend(image_inputs)
    
    # Batch process
    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate captions
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    captions = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return captions


def generate_captions_for_directory(
    image_dir,
    output_json_path,
    model_type="qwen",
    model_path=None,
    prompt_template=None,
    max_new_tokens=50,
    batch_size=1,
    resume=True,
    save_freq=100,
    device_map="auto",
    image_extensions=None
):
    """
    Generate captions for all images in a directory and save to JSON.
    
    Supports both BLIP and Qwen3VL models.
    
    Args:
        image_dir (str): Directory containing images.
        output_json_path (str): Path to save the output JSON file.
        model_type (str): Type of model to use: "blip" or "qwen". Default: "qwen"
        model_path (str): Path to model. If None, uses default for model_type.
                         BLIP default: Salesforce/blip-image-captioning-large
                         Qwen default: Qwen/Qwen3-VL-2B-Instruct
        prompt_template (str): Custom prompt template for caption generation.
        max_new_tokens (int): Maximum tokens to generate per caption.
                             Default is 50 to stay well under SD 2.1's 77 token limit.
        batch_size (int): Number of images to process in one batch. 
                         Larger batches = faster processing but more VRAM.
        resume (bool): If True, resume from existing JSON file.
        save_freq (int): Save progress every N images (default: 100).
        device_map (str): Device mapping for multi-GPU (Qwen) or device for BLIP.
        image_extensions (list): List of valid image extensions
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # Set default model paths
    if model_path is None:
        if model_type == "blip":
            model_path = "Salesforce/blip-image-captioning-large"
        else:
            model_path = "Qwen/Qwen3-VL-2B-Instruct"
    
    # Print GPU info
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"Model Type: {model_type.upper()}")
    print(f"GPU Information:")
    print(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Batch size: {batch_size}")
    if model_type == "qwen":
        print(f"  Device map: {device_map}")
    else:
        print(f"  Device: {device_map if device_map != 'auto' else 'cuda'}")
    print(f"{'='*60}\n")
    
    # Load existing captions if resuming
    existing_captions = {}
    if resume and os.path.exists(output_json_path):
        print(f"Resuming from existing file: {output_json_path}")
        with open(output_json_path, 'r') as f:
            existing_captions = json.load(f)
        print(f"Found {len(existing_captions)} existing captions")
    
    # Get all image files
    image_files = []
    for f in sorted(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, f)
        if os.path.isfile(file_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_files.append(f)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Filter out already processed images
    if resume:
        images_to_process = [f for f in image_files if f not in existing_captions]
        print(f"Need to process {len(images_to_process)} new images")
    else:
        images_to_process = image_files
    
    if len(images_to_process) == 0:
        print("No new images to process!")
        return
    
    # Load model based on type
    if model_type == "blip":
        device = device_map if device_map != "auto" else "cuda"
        model, processor = load_blip_model(model_path, device=device)
        caption_func = generate_caption_blip
        batch_caption_func = generate_captions_batch_blip
    else:  # qwen
        model, processor = load_qwen3vl_model(model_path, device_map=device_map)
        caption_func = generate_caption
        batch_caption_func = generate_captions_batch
    
    # Generate captions
    captions = existing_captions.copy()
    
    try:
        # Process in batches
        num_processed = 0
        for i in tqdm(range(0, len(images_to_process), batch_size), desc="Generating captions"):
            batch_names = images_to_process[i:i + batch_size]
            batch_paths = [os.path.join(image_dir, name) for name in batch_names]
            
            try:
                if batch_size == 1:
                    # Single image processing
                    caption = caption_func(
                        batch_paths[0],
                        model,
                        processor,
                        prompt_template=prompt_template,
                        max_new_tokens=max_new_tokens
                    )
                    batch_captions = [caption]
                else:
                    # Batch processing
                    batch_captions = batch_caption_func(
                        batch_paths,
                        model,
                        processor,
                        prompt_template=prompt_template,
                        max_new_tokens=max_new_tokens
                    )
                
                # Store captions
                for img_name, caption in zip(batch_names, batch_captions):
                    captions[img_name] = caption
                    num_processed += 1
                
                # Save periodically based on save_freq
                if num_processed % save_freq < batch_size:
                    with open(output_json_path, 'w') as f:
                        json.dump(captions, f, indent=2, ensure_ascii=False)
                    tqdm.write(f"Progress saved ({len(captions)} captions)")
                
            except Exception as e:
                tqdm.write(f"\nError processing batch starting with {batch_names[0]}: {e}")
                continue
        
        # Final save to ensure all captions are saved
        with open(output_json_path, 'w') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        with open(output_json_path, 'w') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(captions)} captions to {output_json_path}")
        return
    
    print(f"\nCompleted! Saved {len(captions)} captions to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using BLIP or Qwen3VL")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save output JSON file with captions"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen",
        choices=["blip", "qwen"],
        help="Type of model to use: 'blip' or 'qwen' (default: qwen)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model or HuggingFace model name. If not specified, uses default for model_type. "
             "BLIP default: Salesforce/blip-image-captioning-large, Qwen default: Qwen/Qwen3-VL-2B-Instruct"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt template for caption generation. "
             "For Qwen: default is 'Describe this image in one short sentence.' "
             "For BLIP: optional conditional generation prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate per caption (default: 50 to stay under SD 2.1's 77 token limit)"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start from scratch instead of resuming from existing JSON"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Save progress every N images (default: 100). Set to 1 to save after each image."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images (default: 1). Larger batches are faster but use more VRAM."
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy. For Qwen: 'auto' (recommended), 'balanced', 'sequential', or 'cuda:0'. "
             "For BLIP: 'cuda' or 'cuda:0' etc."
    )
    
    args = parser.parse_args()
    
    generate_captions_for_directory(
        image_dir=args.image_dir,
        output_json_path=args.output_json,
        model_type=args.model_type,
        model_path=args.model_path,
        prompt_template=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        save_freq=args.save_freq,
        device_map=args.device_map
    )


if __name__ == "__main__":
    main()

