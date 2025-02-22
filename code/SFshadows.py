import os
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter

def create_radial_gradient_mask(size, center=None, radius=None, max_intensity=150):
    """
    Creates a radial gradient mask where the intensity increases from the center
    (transparent) to the edges (darker).
    """
    width, height = size
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(width, height) // 2

    # Create coordinate grids
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Compute distance from the center
    distance = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    
    # Normalize and scale to max_intensity (0 means no shadow, max_intensity means full shadow)
    normalized = np.clip(distance / radius, 0, 1)
    alpha = (normalized * max_intensity).astype(np.uint8)
    
    return Image.fromarray(alpha, mode='L')

def create_linear_gradient_mask(size, shadow_side="left", full_shadow_fraction=0.5, max_intensity=150):
    """
    Creates a linear gradient mask for a shadow that is cast from one side.
    
    Parameters:
      - size: (width, height) of the image.
      - shadow_side: one of "left", "right", "top", or "bottom". This determines
        which side of the card has the full shadow.
      - full_shadow_fraction: fraction of the image width (or height) that is at full
        intensity. Beyond that, the intensity fades to 0.
      - max_intensity: maximum alpha (0–255) for the shadow.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if shadow_side in ["left", "right"]:
        full_width = int(width * full_shadow_fraction)
        if shadow_side == "left":
            # Full shadow on the left portion then fade out toward the right.
            mask[:, :full_width] = max_intensity

            if width - full_width > 0:
                fade = np.linspace(max_intensity, 0, width - full_width, endpoint=True)
                mask[:, full_width:] = np.tile(fade, (height, 1))
        else:  # "right"
            mask[:, width - full_width:] = max_intensity
            if width - full_width > 0:
                fade = np.linspace(0, max_intensity, width - full_width, endpoint=True)
                mask[:, :width - full_width] = np.tile(fade, (height, 1))
    
    elif shadow_side in ["top", "bottom"]:
        full_height = int(height * full_shadow_fraction)
        if shadow_side == "top":
            mask[:full_height, :] = max_intensity
            if height - full_height > 0:
                fade = np.linspace(max_intensity, 0, height - full_height, endpoint=True)
                mask[full_height:, :] = np.tile(fade.reshape(-1, 1), (1, width))
        else:  # "bottom"
            mask[height - full_height:, :] = max_intensity
            if height - full_height > 0:
                fade = np.linspace(0, max_intensity, height - full_height, endpoint=True)
                mask[:height - full_height, :] = np.tile(fade.reshape(-1, 1), (1, width))
    else:
        raise ValueError("shadow_side must be one of 'left', 'right', 'top', or 'bottom'.")
    
    return Image.fromarray(mask, mode='L')

def apply_shadow(image, shadow_type="radial", **kwargs):
    """
    Applies a shadow overlay on top of the image.
    
    shadow_type options:
      - "radial": Uses a circular gradient (as if under a canopy).
      - "linear": Uses a linear gradient to simulate a shadow cast on one side.
    
    Additional keyword arguments are passed to the respective mask creation functions.
    """
    image = image.convert("RGBA")
    width, height = image.size
    
    if shadow_type == "radial":
        center = kwargs.get("center", (width // 2, height // 2))
        radius = kwargs.get("radius", min(width, height) // 2)
        max_intensity = kwargs.get("max_intensity", 150)
        blur_radius = kwargs.get("blur_radius", 10)
        
        mask = create_radial_gradient_mask((width, height), center, radius, max_intensity)
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        
    elif shadow_type == "linear":
        shadow_side = kwargs.get("shadow_side", "left")
        full_shadow_fraction = kwargs.get("full_shadow_fraction", 0.5)
        max_intensity = kwargs.get("max_intensity", 150)
        blur_radius = kwargs.get("blur_radius", 10)
        
        mask = create_linear_gradient_mask((width, height), shadow_side, full_shadow_fraction, max_intensity)
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    else:
        raise ValueError("Unsupported shadow_type. Options are: 'radial', 'linear', 'custom'.")
    
    # Create a black overlay using the mask for the alpha channel.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay.putalpha(mask)
    
    shaded_image = Image.alpha_composite(image, overlay)
    return shaded_image

def process_image(image_path, output_path):
    """
    Processes a single image by applying a random shadow effect.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return

    width, height = img.size

    # Randomly choose a shadow type.
    shadow_types = ["radial", "linear"]
    shadow_type = random.choice(shadow_types)

    # Set random parameters for the chosen shadow type.
    if shadow_type == "radial":
        # Choose a random center somewhere on the image.
        center = (random.randint(0, width), random.randint(0, height))
        # Choose a random radius.
        radius = random.randint(min(width, height) // 4, min(width, height) // 2)
        max_intensity = random.randint(100, 200)   # Shadow intensity between 100 and 200.
        blur_radius = random.randint(5, 20)
        img_out = apply_shadow(
            img,
            shadow_type="radial",
            center=center,
            radius=radius,
            max_intensity=max_intensity,
            blur_radius=blur_radius
        )
    elif shadow_type == "linear":
        shadow_side = random.choice(["left", "right", "top", "bottom"])
        full_shadow_fraction = random.uniform(0.3, 0.7)
        max_intensity = random.randint(100, 200)
        blur_radius = random.randint(5, 20)
        img_out = apply_shadow(
            img,
            shadow_type="linear",
            shadow_side=shadow_side,
            full_shadow_fraction=full_shadow_fraction,
            max_intensity=max_intensity,
            blur_radius=blur_radius
        )
    else:
        # Fallback: if for some reason shadow_type is not set
        img_out = img

    try:
        img_out.save(output_path)
        print(f"Processed: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

def process_directory(input_dir, output_dir):
    """
    Processes all images in the input directory and saves the results in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process only image files (customize extensions as needed)
    valid_extensions = (".jpg", ".jpeg")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_files)} images in {input_dir}.")

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        process_image(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images to add random shadows.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save processed images")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

