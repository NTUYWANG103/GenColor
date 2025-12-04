import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Union


def create_checkerboard(width, height, square_size=20):
    """Create a checkerboard pattern background."""
    checkerboard = Image.new('RGB', (width, height))
    pixels = checkerboard.load()
    
    for y in range(height):
        for x in range(width):
            # Determine if this pixel should be light or dark
            is_light = ((x // square_size) + (y // square_size)) % 2 == 0
            color = (200, 200, 200) if is_light else (150, 150, 150)
            pixels[x, y] = color
    
    return checkerboard


def crop_and_magnify(
    image: Union[np.ndarray, Image.Image],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    magnify_scale: float = 2.0,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    dash_length: int = 10,
    box_width: int = 5,
    concat_direction: str = 'horizontal'
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Crop a region from an image and create visualization images.
    
    Args:
        image: Input image (numpy array or PIL Image)
        top_left: (x, y) coordinates of top-left corner
        bottom_right: (x, y) coordinates of bottom-right corner
        magnify_scale: Scale factor for the magnified crop in the combined view
        box_color: RGB color for the dashed box (default: red)
        dash_length: Length of dashes in the box outline
        box_width: Width of the box lines (default: 5)
        concat_direction: Direction to concatenate images ('horizontal' or 'vertical')
        
    Returns:
        Tuple of three PIL Images:
        - image_with_box: Original image with red dashed box
        - cropped_image: Cropped region
        - combined_image: Original and magnified crop concatenated
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        # Convert BGR to RGB if needed (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Ensure coordinates are within image bounds
    width, height = pil_image.size
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    
    # 1. Original image with red dashed box
    image_with_box = pil_image.copy()
    draw = ImageDraw.Draw(image_with_box)
    
    # Draw dashed rectangle
    def draw_dashed_line(draw, start, end, fill, width=2, dash_length=10):
        """Draw a dashed line between two points."""
        x0, y0 = start
        x1, y1 = end
        
        # Calculate line length and direction
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Draw dashes
        current_length = 0
        while current_length < length:
            # Start of dash
            start_x = x0 + dx * current_length
            start_y = y0 + dy * current_length
            
            # End of dash
            end_length = min(current_length + dash_length, length)
            end_x = x0 + dx * end_length
            end_y = y0 + dy * end_length
            
            draw.line([(start_x, start_y), (end_x, end_y)], fill=fill, width=width)
            
            # Move to next dash (skip gap)
            current_length += dash_length * 2
    
    # Draw all four sides of the rectangle with dashes
    draw_dashed_line(draw, (x1, y1), (x2, y1), box_color, width=box_width, dash_length=dash_length)  # Top
    draw_dashed_line(draw, (x2, y1), (x2, y2), box_color, width=box_width, dash_length=dash_length)  # Right
    draw_dashed_line(draw, (x2, y2), (x1, y2), box_color, width=box_width, dash_length=dash_length)  # Bottom
    draw_dashed_line(draw, (x1, y2), (x1, y1), box_color, width=box_width, dash_length=dash_length)  # Left
    
    # 2. Cropped image
    cropped_image = pil_image.crop((x1, y1, x2, y2))
    
    # 3. Combined image: original with box + magnified crop
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Resize cropped image for magnification
    magnified_width = int(crop_width * magnify_scale)
    magnified_height = int(crop_height * magnify_scale)
    magnified_crop = cropped_image.resize((magnified_width, magnified_height), Image.LANCZOS)
    
    # Create combined image with checkerboard background
    padding = 20
    
    if concat_direction.lower() == 'horizontal':
        # Horizontal concatenation (side by side)
        combined_width = width + padding + magnified_width
        combined_height = max(height, magnified_height)
        
        # Create checkerboard background
        combined_image = create_checkerboard(combined_width, combined_height)
        
        # Paste original image with box on the left
        combined_image.paste(image_with_box, (0, 0))
        
        # Paste magnified crop on the right
        combined_image.paste(magnified_crop, (width + padding, 0))
        
        # Draw connection lines from box to magnified region
        draw_combined = ImageDraw.Draw(combined_image)
        # Top-left corner connection (dashed)
        draw_dashed_line(draw_combined, (x1, y1), (width + padding, 0), box_color, width=box_width, dash_length=dash_length)
        # Top-right corner connection (dashed)
        draw_dashed_line(draw_combined, (x2, y1), (width + padding + magnified_width, 0), box_color, width=box_width, dash_length=dash_length)
    
    elif concat_direction.lower() == 'vertical':
        # Vertical concatenation (top to bottom)
        combined_width = max(width, magnified_width)
        combined_height = height + padding + magnified_height
        
        # Create checkerboard background
        combined_image = create_checkerboard(combined_width, combined_height)
        
        # Paste original image with box on the top
        combined_image.paste(image_with_box, (0, 0))
        
        # Paste magnified crop on the bottom
        combined_image.paste(magnified_crop, (0, height + padding))
        
        # Draw connection lines from box to magnified region
        draw_combined = ImageDraw.Draw(combined_image)
        # Top-left corner connection (dashed)
        draw_dashed_line(draw_combined, (x1, y1), (0, height + padding), box_color, width=box_width, dash_length=dash_length)
        # Top-right corner connection (dashed)
        draw_dashed_line(draw_combined, (x2, y1), (magnified_width, height + padding), box_color, width=box_width, dash_length=dash_length)
    
    else:
        raise ValueError(f"concat_direction must be 'horizontal' or 'vertical', got '{concat_direction}'")
    
    return image_with_box, cropped_image, combined_image


# Example usage
if __name__ == "__main__":
    # Load an example image
    image_path = "example.jpg"  # Replace with your image path
    image = Image.open(image_path)
    
    # Define crop region (top-left and bottom-right coordinates)
    top_left = (100, 100)
    bottom_right = (300, 300)
    
    # Get the three images
    img_with_box, cropped, combined = crop_and_magnify(
        image, 
        top_left, 
        bottom_right,
        magnify_scale=2.5
    )
    
    # Save or display the results
    img_with_box.save("output_with_box.png")
    cropped.save("output_cropped.png")
    combined.save("output_combined.png")
    
    print("Images saved successfully!")
    print(f"Original size: {image.size}")
    print(f"Cropped size: {cropped.size}")
    print(f"Combined size: {combined.size}")

