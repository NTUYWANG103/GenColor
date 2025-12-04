import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pillow_lut import rgb_color_enhance
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000

def img_resize(img, target_shorter_side):
    width, height = img.size
    if width <= height:
        # width is shorter side
        new_width = target_shorter_side
        new_height = int(height * (new_width / width))
    else:
        # height is shorter side
        new_height = target_shorter_side
        new_width = int(width * (new_height / height))

    return img.resize((new_width, new_height))

def random_square_crop(img, target_size, bbox=None):
    """
    Randomly (or deterministically if bbox is given) crop out a square 
    of `target_size` x `target_size` from the given PIL image `img`.
    
    Args:
        img (PIL.Image): The input image (already resized by `img_resize`).
        target_size (int): The desired size for the square crop.
        bbox (tuple, optional): If provided, use this bounding box 
            (left, top, right, bottom) to crop. If None, randomly choose one.

    Returns:
        cropped_img (PIL.Image): The cropped image of size (target_size x target_size).
        bbox (tuple): The (left, top, right, bottom) bounding box used for cropping.
    """
    width, height = img.size

    # If width or height is still < target_size, it means we cannot crop a target_size square.
    # You can decide how to handle this. Here, we force it to match exactly by minimal upscaling.
    if width < target_size or height < target_size:
        scale = target_size / min(width, height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        width, height = new_w, new_h

    if bbox is None:
        # Choose a random bounding box of size `target_size` x `target_size`.
        left = random.randint(0, width - target_size)
        top = random.randint(0, height - target_size)
        right = left + target_size
        bottom = top + target_size
        bbox = (left, top, right, bottom)
    else:
        # Use the provided bounding box (for alignment with another image).
        left, top, right, bottom = bbox

    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img, bbox

def generate_random_lut(choose_rate=0.5, param_config=None, min_sample=1, std_scale=0.1):
    """
    Generate a random LUT using normal distribution sampling.
    
    Args:
        choose_rate: Probability of choosing each parameter
        param_config: Dictionary with parameter settings (range, mean, std_scale)
                     If std_scale is provided per-param, it overrides the global std_scale
        min_sample: Minimum number of parameters to select
        std_scale: Global scale factor for std. std = range_size * std_scale
                  Default 0.3 means std covers ~30% of the range from mean.
                  With normal distribution, ~68% samples within 1*std, ~95% within 2*std.
    """
    if param_config is None:
        # Full ranges from the documentation with defaults as mean
        # std_scale can be customized per parameter if needed
        param_config = {
            'brightness': {'range': (-1.0, 1.0), 'mean': 0.0},
            'exposure': {'range': (-5.0, 5.0), 'mean': 0.0},
            'contrast': {'range': (-1.0, 5.0), 'mean': 0.0},
            'warmth': {'range': (-1.0, 1.0), 'mean': 0.0},
            'saturation': {'range': (-1.0, 5.0), 'mean': 0.0},
            'vibrance': {'range': (-1.0, 5.0), 'mean': 0.0},
            'hue': {'range': (0.0, 1.0), 'mean': 0.0},  # Custom: hue is more sensitive
            'gamma': {'range': (0.0, 10.0), 'mean': 1.0},
        }
    
    params = {}
    
    # Initially select parameters based on choose_rate
    for param, config in param_config.items():
        if random.random() < choose_rate:
            # Calculate std based on range size and std_scale
            range_size = config['range'][1] - config['range'][0]
            param_std_scale = config.get('std_scale', std_scale)
            std = range_size * param_std_scale
            
            # Sample from normal distribution
            value = np.random.normal(config['mean'], std)
            # Clip to valid range
            value = np.clip(value, config['range'][0], config['range'][1])
            params[param] = float(value)
    
    # Ensure at least min_sample parameters are selected
    while len(params) < min_sample:
        param_to_add = random.choice(list(param_config.keys()))
        if param_to_add not in params:
            config = param_config[param_to_add]
            range_size = config['range'][1] - config['range'][0]
            param_std_scale = config.get('std_scale', std_scale)
            std = range_size * param_std_scale
            
            value = np.random.normal(config['mean'], std)
            value = np.clip(value, config['range'][0], config['range'][1])
            params[param_to_add] = float(value)
    
    return rgb_color_enhance(16, **params)

def calculate_color_metrics(original_img, transformed_img, metrics=['ssim', 'delta_e']):
    """
    Calculate color difference metrics between two images.
    
    Args:
        original_img: Original PIL Image
        transformed_img: Transformed PIL Image
        metrics: List of metric names to calculate. Available: 'ssim', 'delta_e'
    
    Returns:
        dict: Dictionary containing requested metrics
    """
    results = {}
    original_array = np.array(original_img).astype(np.float32)
    transformed_array = np.array(transformed_img).astype(np.float32)
    
    # SSIM - Structural Similarity (higher is better, range 0-1)
    if 'ssim' in metrics:
        ssim_score = ssim(original_array.astype(np.uint8), transformed_array.astype(np.uint8), 
                          channel_axis=2, data_range=255)
        results['ssim'] = ssim_score
    
    # Delta E - Perceptual color difference in LAB space (lower is better)
    if 'delta_e' in metrics:
        # Convert to LAB color space (range 0-1 for input)
        original_lab = rgb2lab(original_array / 255.0)
        transformed_lab = rgb2lab(transformed_array / 255.0)
        delta_e = np.mean(deltaE_ciede2000(original_lab, transformed_lab))
        results['delta_e'] = delta_e
    
    return results

def generate_filtered_lut(original_img, choose_rate=0.5, param_config=None, min_sample=1, 
                          std_scale=0.1, delta_e_threshold=15.0, ssim_threshold=0.75, 
                          max_attempts=50):
    """
    Generate a random LUT that satisfies color metric constraints.
    
    Args:
        original_img: The original PIL Image to test against
        choose_rate: Probability of choosing each parameter
        param_config: Dictionary with parameter settings
        min_sample: Minimum number of parameters to select
        std_scale: Global scale factor for std
        delta_e_threshold: Maximum allowed Delta E value (default: 15.0)
        ssim_threshold: Minimum allowed SSIM value (default: 0.75)
        max_attempts: Maximum number of attempts to generate a valid LUT
    
    Returns:
        tuple: (lut, metrics_dict) or (None, None) if no valid LUT found
    """
    for attempt in range(max_attempts):
        # Generate a random LUT
        lut = generate_random_lut(choose_rate, param_config, min_sample, std_scale)
        
        # Apply LUT to create hint image
        hint = original_img.filter(lut)
        
        # Calculate metrics
        metrics = calculate_color_metrics(original_img, hint, metrics=['ssim', 'delta_e'])
        
        # Check if metrics satisfy thresholds
        if metrics['delta_e'] < delta_e_threshold and metrics['ssim'] > ssim_threshold:
            return lut, metrics
    
    # If no valid LUT found after max_attempts, return None
    # In practice, you might want to return the best attempt or use a fallback
    return None, None

class GenColorDataset(Dataset):
    """
    A dataset that loads:
      - The main image from `img_dir`.
      - A corresponding 'hint' image from `hint_dir`.
      - A prompt string from the JSON file `prompt_json_path` 
        that is keyed by the image's base name (file name without extension).
    """
    def __init__(self, img_dir, prompt_json_path, img_size=512, resize_before_crop=True, random_crop=True,
                 use_filtered_lut=True, delta_e_threshold=15.0, ssim_threshold=0.75, 
                 lut_choose_rate=0.5, lut_std_scale=0.1):
        """
        Args:
            img_dir (str): Directory with main images.
            hint_dir (str): Directory with hint images (same filenames as main).
            prompt_json_path (str): Path to JSON with {image_name: prompt}.
            img_size (int): Final random crop size (width=height=img_size).
            use_filtered_lut (bool): Whether to use filtered LUT generation with metric constraints.
            delta_e_threshold (float): Maximum allowed Delta E value (default: 15.0).
            ssim_threshold (float): Minimum allowed SSIM value (default: 0.75).
            lut_choose_rate (float): Probability of choosing each LUT parameter (default: 0.5).
            lut_std_scale (float): Standard deviation scale for LUT parameters (default: 0.1).
        """
        self.img_dir = img_dir
        self.prompt_json_path = prompt_json_path
        self.img_size = img_size
        self.resize_before_crop = resize_before_crop
        self.random_crop = random_crop
        self.use_filtered_lut = use_filtered_lut
        self.delta_e_threshold = delta_e_threshold
        self.ssim_threshold = ssim_threshold
        self.lut_choose_rate = lut_choose_rate
        self.lut_std_scale = lut_std_scale
        
        # Load the JSON file that maps each image base name to a prompt
        with open(self.prompt_json_path, 'r') as f:
            self.prompts = json.load(f)

        # Gather all files from the directory (no extension filtering)
        # self.images = sorted(os.listdir(self.img_dir))
        # using keys of the json file
        self.images = sorted(list(self.prompts.keys()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Look up the prompt; return empty string if missing
        prompt = self.prompts.get(img_name, "")

        # --- Load the main image ---
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # --- Generate hint image using filtered or random LUT ---
        if self.use_filtered_lut:
            # Use filtered LUT generation with metric constraints
            random_lut, metrics = generate_filtered_lut(
                img, 
                choose_rate=self.lut_choose_rate,
                std_scale=self.lut_std_scale,
                delta_e_threshold=self.delta_e_threshold,
                ssim_threshold=self.ssim_threshold
            )
            
            # Fallback to simple random LUT if filtering fails
            if random_lut is None:
                print(f"Filtered LUT generation failed, using simple random LUT.")
                random_lut = generate_random_lut(
                    choose_rate=self.lut_choose_rate,
                    std_scale=self.lut_std_scale
                )
        else:
            # Use simple random LUT without filtering
            random_lut = generate_random_lut(
                choose_rate=self.lut_choose_rate,
                std_scale=self.lut_std_scale
            )
        
        hint = img.filter(random_lut)

        # 1) Resize both images so the shorter side is `img_size`.
        if self.resize_before_crop:
            img = img_resize(img, self.img_size)
            hint = img_resize(hint, self.img_size)

        # 2) Random-crop both images to exactly (img_size x img_size) using the same bounding box.
        if self.random_crop:
            cropped_img, bbox = random_square_crop(img, self.img_size)
            cropped_hint, _ = random_square_crop(hint, self.img_size, bbox=bbox)

        # 3) Convert to tensor in [-1, 1]
        img_np = np.array(cropped_img).astype(np.float32)
        img_tensor = torch.from_numpy((img_np / 127.5) - 1.0).permute(2, 0, 1)

        hint_np = np.array(cropped_hint).astype(np.float32)
        hint_tensor = torch.from_numpy((hint_np / 255.0)).permute(2, 0, 1) # different from flux version, output 0-1

        return {'pixel_values': img_tensor, 'conditioning_pixel_values': hint_tensor, 'prompt': prompt, 'img_path': img_path}


def gencolor_loader(batch_size, num_workers, shuffle=True, **dataset_kwargs):
    """
    Helper function to create a DataLoader for GenColorDataset.

    Args:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of worker processes.
        shuffle (bool): Whether to shuffle the dataset.
        **dataset_kwargs: Additional arguments passed to GenColorDataset constructor.

    Returns:
        DataLoader: A DataLoader instance for the GenColorDataset.
    """
    dataset = GenColorDataset(**dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)