"""
ComfyUI node for GenColor Fusion
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import folder_paths

# Add local colorfusion module
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'colorfusion'))

from colorfusion_utils import load_colorfusion_model, gencolor_forward, gencolor_feed_data, tensor2img

# Register model folder
folder_paths.folder_names_and_paths["gencolor"] = (
    [os.path.join(folder_paths.models_dir, "gencolor")],
    folder_paths.supported_pt_extensions
)


class GenColorFusion:
    """GenColor Fusion - texture-preserving color transfer"""

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "color_reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("fused_image",)
    FUNCTION = "apply_fusion"
    CATEGORY = "GenColor"

    def apply_fusion(self, input_image, color_reference):
        # Load model (cached)
        if "model" not in self._model_cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Find model path
            model_path = folder_paths.get_full_path("gencolor", "base.pth")
            if model_path is None:
                raise FileNotFoundError("Model not found. Please put base.pth in ComfyUI/models/gencolor/")

            model = load_colorfusion_model(model_path, in_chans=6, device=device, model_type="base")
            model = model.to(torch.float32).eval()
            self._model_cache["model"] = model
            self._model_cache["device"] = device

        model = self._model_cache["model"]
        device = self._model_cache["device"]

        # Process
        batch_size = input_image.shape[0]
        results = []

        for i in range(batch_size):
            input_np = (input_image[i].cpu().numpy() * 255).astype(np.uint8)
            input_pil = Image.fromarray(input_np)

            ref_idx = min(i, color_reference.shape[0] - 1)
            ref_np = (color_reference[ref_idx].cpu().numpy() * 255).astype(np.uint8)
            ref_pil = Image.fromarray(ref_np).resize(input_pil.size, Image.Resampling.LANCZOS)

            data_tensor = gencolor_feed_data(input_pil, ref_pil, device=device)
            for key, value in data_tensor.items():
                if isinstance(value, torch.Tensor):
                    data_tensor[key] = value.to(torch.float32)

            with torch.no_grad():
                fusion_tensor = gencolor_forward(model, data_tensor, return_np=False)

            fusion_np = tensor2img(fusion_tensor, rgb2bgr=False)
            result_np = np.array(fusion_np).astype(np.float32) / 255.0
            results.append(result_np)

        output = torch.from_numpy(np.stack(results, axis=0))
        return (output,)


NODE_CLASS_MAPPINGS = {
    "GenColorFusion": GenColorFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenColorFusion": "GenColor Fusion",
}
