
import os
import sys
import argparse
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageOps

# Add model paths
sys.path.append('model_banks/colorfusion')
from colorfusion_utils import load_colorfusion_model, gencolor_forward, gencolor_feed_data
sys.path.append('model_banks/Harmonizer/src')
from inference_harmonizer_api import EnhancerAPI


def apply_equalization(img_pil, cutoff=0):
    """Apply histogram equalization to the image."""
    return ImageOps.autocontrast(img_pil, cutoff=cutoff, preserve_tone=True)


class GenColorPipeline:
    """GenColor image enhancement pipeline."""
    
    def __init__(
        self,
        controlnet_paths=None,
        conditioning_scales=None,
        fusion_model_path="ckpt/GenColor_fusion/base.pth",
        harmonizer_path="model_banks/Harmonizer/pretrained/enhancer.pth",
        device=None,
        dtype="bfloat16",
        compile_fusion=False
    ):
        """
        Initialize the GenColor pipeline.
        
        Args:
            controlnet_paths: List of paths to ControlNet models
            conditioning_scales: List of conditioning scales for each ControlNet
            fusion_model_path: Path to the fusion model
            harmonizer_path: Path to the harmonizer model
            device: Device to run on (cuda/cpu)
            dtype: Data type for inference ('float32', 'float16', 'bfloat16')
            compile_fusion: Whether to compile fusion model with torch.compile
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype.lower(), torch.bfloat16)
        self.dtype_str = dtype.lower()
        
        # Enable optimizations for faster inference
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        
        # Default to combine_weight setting from bash script
        if controlnet_paths is None:
            self.controlnet_paths = [
                "ckpt/GenColor_SD21/mid/controlnet",
                "ckpt/GenColor_SD21/late/controlnet"
            ]
        if conditioning_scales is None:
            conditioning_scales = [0.5, 0.5]
        
        self.conditioning_scales = conditioning_scales
        
        print("Loading ControlNet models...")
        self.controlnets = [ControlNetModel.from_pretrained(path) for path in self.controlnet_paths]
        
        print(f"Loading Stable Diffusion pipeline (dtype: {self.dtype_str})...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=self.controlnets,
            torch_dtype=self.dtype
        )
        self.pipe = self.pipe.to(self.dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=False)
        
        print(f"Loading fusion model (dtype: {self.dtype_str})...")
        self.fusion_model = load_colorfusion_model(
            load_path=fusion_model_path,
            in_chans=6,
            device=self.device,
            model_type='base'
        )
        self.fusion_model = self.fusion_model.to(self.dtype).eval()
        
        # Compile fusion model for faster inference
        if compile_fusion and hasattr(torch, 'compile'):
            print("Compiling fusion model with torch.compile...")
            self.fusion_model = torch.compile(self.fusion_model, mode='reduce-overhead')
        
        print("Loading harmonizer...")
        self.harmonizer = EnhancerAPI(pretrained_path=harmonizer_path)
        
        print("Pipeline initialized successfully!")
    
    def process(
        self,
        input_img_path,
        output_path,
        prompt="",
        negative_prompt="low quality, bad quality, low contrast, low saturation, dark, black and white, color bleeding, bw, monochrome, grainy, blurry, historical, restored, desaturate",
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=0,
        cutoff=0
    ):
        """
        Process a single image through the GenColor pipeline.
        
        Args:
            input_img_path: Path to input image
            output_path: Path to save output image
            prompt: Text prompt for generation (default: empty)
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for diffusion
            seed: Random seed
            cutoff: Cutoff for histogram equalization
        
        Returns:
            PIL Image: Final enhanced image
        """
        print(f"\nProcessing: {input_img_path}")
        
        # Load and preprocess input image
        print("Loading input image...")
        input_img = Image.open(input_img_path).convert('RGB')
        input_img_equalized = apply_equalization(input_img, cutoff=cutoff)
        
        # Generate color reference using diffusion model
        print("Generating color reference with diffusion model...")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            diffu_gen = self.pipe(
                prompt=[prompt],
                negative_prompt=[negative_prompt],
                image=[input_img_equalized] * len(self.controlnets), # can downsample for faster inference
                controlnet_conditioning_scale=self.conditioning_scales,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=[generator]
            ).images
        
        # Apply equalization to diffusion output
        color_ref_img = apply_equalization(diffu_gen[0], cutoff=cutoff)
        
        # Fusion
        print("Fusing input with color reference...")
        with torch.no_grad():
            data_tensor = gencolor_feed_data(input_img_equalized, color_ref_img, device=self.device)
            # Convert data to the same dtype as model
            for key, value in data_tensor.items():
                if isinstance(value, torch.Tensor):
                    data_tensor[key] = value.to(self.dtype)
            robust_fusion = gencolor_forward(self.fusion_model, data_tensor)
        
        fusion_img = Image.fromarray(robust_fusion)
        fusion_img_equalized = apply_equalization(fusion_img, cutoff=cutoff)
        
        # Harmonization
        print("Applying filter...")
        final_output = self.harmonizer.inference(fusion_img_equalized)
        final_output_equalized = apply_equalization(final_output, cutoff=cutoff)
        
        # Save result
        print(f"Saving result to: {output_path}")
        final_output_equalized.save(output_path)
        
        print("Done!")
        return final_output_equalized


def main():
    parser = argparse.ArgumentParser(description="GenColor: Image Enhancement Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image path")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (default: empty)")
    parser.add_argument("--negative_prompt", type=str, 
                        default="low quality, bad quality, low contrast, low saturation, dark, black and white, color bleeding, bw, monochrome, grainy, blurry, historical, restored, desaturate",
                        help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cutoff", type=int, default=0, help="Histogram equalization cutoff")
    parser.add_argument("--controlnet_path", type=str, nargs='+', default=None, 
                        help="Paths to ControlNet models")
    parser.add_argument("--controlnet_scale", type=float, nargs='+', default=None,
                        help="Conditioning scales for ControlNets")
    parser.add_argument("--fusion_model_path", type=str, default="ckpt/GenColor_fusion/base.pth",
                        help="Path to fusion model (default: ckpt/GenColor_fusion/base.pth)")
    parser.add_argument("--harmonizer_path", type=str, default="model_banks/Harmonizer/pretrained/enhancer.pth",
                        help="Path to harmonizer model (default: model_banks/Harmonizer/pretrained/enhancer.pth)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        choices=["float32", "float16", "fp16", "bfloat16", "bf16"],
                        help="Data type for inference (default: bfloat16). float16 is ~5x faster than float32 on A100")
    parser.add_argument("--compile_fusion", action="store_true",
                        help="Compile fusion model with torch.compile for faster inference (requires PyTorch 2.0+)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GenColorPipeline(
        controlnet_paths=args.controlnet_path,
        conditioning_scales=args.controlnet_scale,
        fusion_model_path=args.fusion_model_path,
        harmonizer_path=args.harmonizer_path,
        device=args.device,
        dtype=args.dtype,
        compile_fusion=args.compile_fusion
    )
    
    # Process image
    pipeline.process(
        input_img_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        cutoff=args.cutoff
    )


if __name__ == "__main__":
    main()

