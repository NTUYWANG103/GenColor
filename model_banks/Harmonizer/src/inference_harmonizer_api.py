import os
import sys
import torch
import torchvision.transforms.functional as tf
from PIL import Image
import model


class EnhancerAPI:
    """
    Harmonizer API for image enhancement.
    
    This class provides a simple interface to load the Harmonizer model and perform
    image enhancement on PIL Images.
    """
    
    def __init__(self, pretrained_path='./pretrained/harmonizer.pth', apply_filter_mask=None):
        """
        Initialize the Harmonizer model.
        
        Args:
            pretrained_path (str): Path to the pretrained model file
            apply_filter_mask (list): List of 5 boolean values for filters to apply
                                    [brightness, contrast, saturation, hue, color]
                                    Default: [True, True, True, True, True]
        """
        self.pretrained_path = pretrained_path
        self.cuda = torch.cuda.is_available()
        
        # Default filter mask if not provided
        if apply_filter_mask is None:
            self.apply_filter_mask = [True, True, True, True, True]
        else:
            if len(apply_filter_mask) != 5:
                raise ValueError("apply_filter_mask must contain exactly 5 values (got {})".format(len(apply_filter_mask)))
            self.apply_filter_mask = apply_filter_mask
        
        # Initialize and load the model
        self._load_model()
        
    def _load_model(self):
        """Load the harmonizer model from the pretrained weights."""
        print(f"Loading Harmonizer model from: {self.pretrained_path}")
        
        # Create the harmonizer model
        self.harmonizer = model.Enhancer()
        
        if self.cuda:
            self.harmonizer = self.harmonizer.cuda()
            
        # Load pretrained weights
        if not os.path.exists(self.pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {self.pretrained_path}")
            
        self.harmonizer.load_state_dict(torch.load(self.pretrained_path), strict=True)
        self.harmonizer.eval()
        
        print("Harmonizer model loaded successfully!")
        print(f"CUDA available: {self.cuda}")
        print(f"Apply filter mask: {self.apply_filter_mask}")
        
    def inference(self, input_image):
        """
        Perform image harmonization/enhancement on a single image.
        
        Args:
            input_image (PIL.Image): Input image in PIL format
            
        Returns:
            PIL.Image: Enhanced/harmonized image
        """
        if not isinstance(input_image, Image.Image):
            raise TypeError("Input must be a PIL.Image")
            
        # Convert to RGB if needed
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Prepare input tensor
        _comp = tf.to_tensor(input_image)[None, ...]  # shape: (1, 3, H, W)
        _mask = torch.ones(1, 1, _comp.shape[2], _comp.shape[3])  # global mask (all ones)
        
        if self.cuda:
            _comp = _comp.cuda()
            _mask = _mask.cuda()
        
        # Perform inference
        with torch.no_grad():
            # Predict harmonization arguments
            arguments = self.harmonizer.predict_arguments(_comp, _mask)
            
            # Restore/enhance the image
            _harmonized = self.harmonizer.restore_image(
                _comp, _mask, arguments, apply_filter_mask=self.apply_filter_mask
            )[-1]
        
        # Convert back to PIL Image
        result_img = tf.to_pil_image(_harmonized.cpu().squeeze(0).clamp(0, 1))
        
        return result_img
    
    def set_filter_mask(self, apply_filter_mask):
        """
        Update the filter mask for harmonization.
        
        Args:
            apply_filter_mask (list): List of 5 boolean values for filters to apply
                                    [brightness, contrast, saturation, hue, color]
        """
        if len(apply_filter_mask) != 5:
            raise ValueError("apply_filter_mask must contain exactly 5 values (got {})".format(len(apply_filter_mask)))
        self.apply_filter_mask = apply_filter_mask
        print(f"Updated filter mask: {self.apply_filter_mask}")


def create_harmonizer_api(pretrained_path='./pretrained/harmonizer.pth', apply_filter_mask=None):
    """
    Convenience function to create a EnhancerAPI instance.
    
    Args:
        pretrained_path (str): Path to the pretrained model file
        apply_filter_mask (list): List of 5 boolean values for filters to apply
                                [brightness, contrast, saturation, hue, color]
                                Default: [True, True, True, True, True]
    
    Returns:
        EnhancerAPI: Initialized harmonizer API instance
    """
    return EnhancerAPI(pretrained_path=pretrained_path, apply_filter_mask=apply_filter_mask)


# Example usage:
if __name__ == '__main__':
    # Example of how to use the API
    from PIL import Image
    
    # Initialize the API
    harmonizer_api = EnhancerAPI(pretrained_path='../pretrained/enhancer.pth')
    
    # Load an image
    input_image = Image.open('path/to/your/image.jpg')
    
    # Perform harmonization
    enhanced_image = harmonizer_api.inference(input_image)
    
    # Save the result
    enhanced_image.save('path/to/output/enhanced_image.jpg')
    
    print("Image enhancement completed!")
