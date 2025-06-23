"""
Main synthetic data generator for Khmer digits OCR.
"""

import os
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .utils import (
    load_khmer_fonts, validate_font_collection, generate_digit_sequence,
    normalize_khmer_text, create_character_mapping
)
from .backgrounds import BackgroundGenerator
from .augmentation import ImageAugmentor


class SyntheticDataGenerator:
    """
    Generates synthetic training data for Khmer digits OCR.
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str):
        """
        Initialize the synthetic data generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
        """
        self.config_path = config_path
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.image_size = tuple(self.config['model']['input']['image_size'])
        self.max_sequence_length = self.config['model']['characters']['max_sequence_length']
        
        self.background_generator = BackgroundGenerator(self.image_size)
        self.augmentor = ImageAugmentor(
            rotation_range=tuple(self.config['data']['augmentation']['rotation']),
            scale_range=tuple(self.config['data']['augmentation']['scaling']),
            noise_std=self.config['data']['augmentation']['noise']['gaussian_std'],
            brightness_range=tuple(self.config['data']['augmentation']['brightness']),
            contrast_range=tuple(self.config['data']['augmentation']['contrast'])
        )
        
        # Load and validate fonts
        self.fonts = load_khmer_fonts(fonts_dir)
        self.font_validation = validate_font_collection(fonts_dir)
        
        # Filter to only working fonts
        self.working_fonts = {
            name: path for name, path in self.fonts.items() 
            if self.font_validation[name]
        }
        
        if not self.working_fonts:
            raise ValueError("No working Khmer fonts found!")
        
        print(f"Loaded {len(self.working_fonts)} working fonts: {list(self.working_fonts.keys())}")
        
        # Create character mappings
        self.char_to_idx, self.idx_to_char = create_character_mapping()
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_adaptive_font_size(self, text: str) -> int:
        """Get a font size that fits the text within image boundaries."""
        # Start with a reasonable base size
        base_size = int(self.image_size[1] * 0.6)
        
        # Adjust based on text length - longer sequences need smaller fonts
        length_factor = max(0.3, 1.0 - (len(text) - 1) * 0.1)
        target_size = int(base_size * length_factor)
        
        # Test different font sizes to ensure text fits
        for font_size in range(target_size, 12, -2):  # Minimum font size of 12
            # Test with a representative font
            test_font_path = list(self.working_fonts.values())[0]
            test_font = ImageFont.truetype(test_font_path, font_size)
            
            bbox = self._get_text_bbox(text, test_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits with some margin (leave 20% margin on each side)
            margin_width = self.image_size[0] * 0.1
            margin_height = self.image_size[1] * 0.1
            
            if (text_width <= self.image_size[0] - 2 * margin_width and 
                text_height <= self.image_size[1] - 2 * margin_height):
                return font_size
        
        # Fallback to minimum size
        return 12
    
    def _get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        """Get tight bounding box for text."""
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1000, 1000))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        return bbox
    
    def _safe_position_text(self, text: str, font: ImageFont.FreeTypeFont, 
                           image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate safe position for text ensuring it fits within image bounds."""
        bbox = self._get_text_bbox(text, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Define safe margins (10% of image size)
        margin_x = int(image_size[0] * 0.1)
        margin_y = int(image_size[1] * 0.1)
        
        # Calculate available space for text placement
        available_width = image_size[0] - 2 * margin_x
        available_height = image_size[1] - 2 * margin_y
        
        # Center text within available space
        x = margin_x + (available_width - text_width) // 2
        y = margin_y + (available_height - text_height) // 2
        
        # Adjust for bbox offset
        x -= bbox[0]
        y -= bbox[1]
        
        # Ensure text doesn't go outside bounds
        x = max(margin_x - bbox[0], min(x, image_size[0] - margin_x - text_width - bbox[0]))
        y = max(margin_y - bbox[1], min(y, image_size[1] - margin_y - text_height - bbox[1]))
        
        return x, y
    
    def _validate_text_fits(self, text: str, font: ImageFont.FreeTypeFont, 
                           position: Tuple[int, int], image_size: Tuple[int, int]) -> bool:
        """Validate that text fits completely within image boundaries."""
        x, y = position
        bbox = self._get_text_bbox(text, font)
        
        # Calculate actual text boundaries in the image
        text_left = x + bbox[0]
        text_top = y + bbox[1]
        text_right = x + bbox[2]
        text_bottom = y + bbox[3]
        
        # Check if text is within image bounds
        return (text_left >= 0 and text_top >= 0 and 
                text_right <= image_size[0] and text_bottom <= image_size[1])
    
    def generate_single_image(self, 
                             text: Optional[str] = None,
                             font_name: Optional[str] = None,
                             apply_augmentation: bool = True) -> Tuple[Image.Image, Dict]:
        """
        Generate a single synthetic image with text.
        
        Args:
            text: Text to render, if None generates random sequence
            font_name: Font to use, if None chooses random font
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Tuple of (image, metadata)
        """
        # Generate text if not provided
        if text is None:
            text = generate_digit_sequence(1, self.max_sequence_length)
        text = normalize_khmer_text(text)
        
        # Choose font
        if font_name is None:
            font_name = random.choice(list(self.working_fonts.keys()))
        font_path = self.working_fonts[font_name]
        
        # Generate background
        background = self.background_generator.generate_random_background()
        
        # Get optimal text color for this background
        text_color = self.background_generator.get_optimal_text_color(background)
        
        # Create font object with adaptive size
        font_size = self._get_adaptive_font_size(text)
        font = ImageFont.truetype(font_path, font_size)
        
        # Create image with text
        image = background.copy()
        draw = ImageDraw.Draw(image)
        
        # Calculate safe text position
        x, y = self._safe_position_text(text, font, self.image_size)
        
        # Add minimal random offset for variety (but stay within safe bounds)
        max_offset_x = int(self.image_size[0] * 0.02)  # 2% of width
        max_offset_y = int(self.image_size[1] * 0.02)  # 2% of height
        offset_x = random.randint(-max_offset_x, max_offset_x)
        offset_y = random.randint(-max_offset_y, max_offset_y)
        x += offset_x
        y += offset_y
        
        # Validate text fits before drawing
        if not self._validate_text_fits(text, font, (x, y), self.image_size):
            # If text doesn't fit, recalculate with smaller font
            for fallback_size in range(font_size - 2, 8, -2):
                fallback_font = ImageFont.truetype(font_path, fallback_size)
                x, y = self._safe_position_text(text, fallback_font, self.image_size)
                if self._validate_text_fits(text, fallback_font, (x, y), self.image_size):
                    font = fallback_font
                    font_size = fallback_size
                    break
        
        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Apply lighter augmentation to avoid cropping text
        if apply_augmentation:
            # Create a custom augmentor with reduced parameters for better text preservation
            safe_augmentor = ImageAugmentor(
                rotation_range=(-5, 5),      # Reduced rotation to prevent cropping
                scale_range=(0.95, 1.05),    # Minimal scaling
                noise_std=self.augmentor.noise_std,
                brightness_range=self.augmentor.brightness_range,
                contrast_range=self.augmentor.contrast_range
            )
            # Apply only safe augmentations
            safe_augmentations = ['brightness', 'contrast', 'noise']
            if random.random() < 0.3:  # Only occasionally apply rotation
                safe_augmentations.append('rotate')
            
            image = safe_augmentor.augment_image(image, safe_augmentations)
        
        # Ensure final image is correct size
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Create metadata
        metadata = {
            'label': text,
            'font': font_name,
            'font_size': font_size,
            'text_color': list(text_color),  # Convert tuple to list for YAML compatibility
            'text_position': [x, y],         # Convert tuple to list for YAML compatibility
            'sequence_length': len(text),
            'augmented': apply_augmentation
        }
        
        return image, metadata
    
    def generate_dataset(self, 
                        num_samples: int,
                        train_split: float = 0.8,
                        save_images: bool = True,
                        show_progress: bool = True) -> Dict:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        train_dir = Path(self.output_dir) / 'train'
        val_dir = Path(self.output_dir) / 'val'
        
        if save_images:
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []}
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc="Generating dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),  # Convert tuple to list for YAML compatibility
            'max_sequence_length': self.max_sequence_length,
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generated_by': 'SyntheticDataGenerator v1.0'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Fonts used: {len(self.working_fonts)}")
        
        return all_metadata
    
    def generate_samples_by_length(self, 
                                  samples_per_length: int = 100,
                                  save_images: bool = True) -> Dict:
        """
        Generate samples with balanced sequence lengths.
        
        Args:
            samples_per_length: Number of samples per sequence length
            save_images: Whether to save images to disk
            
        Returns:
            Dictionary with dataset information
        """
        all_metadata = {'samples': []}
        
        # Create output directory
        if save_images:
            output_dir = Path(self.output_dir) / 'balanced'
            output_dir.mkdir(exist_ok=True)
        
        sample_count = 0
        
        for length in range(1, self.max_sequence_length + 1):
            print(f"Generating {samples_per_length} samples with {length} digit(s)...")
            
            for i in tqdm(range(samples_per_length), desc=f"Length {length}"):
                # Generate text with specific length
                text = generate_digit_sequence(length, length)
                image, metadata = self.generate_single_image(text=text)
                
                if save_images:
                    image_filename = f"sample_{sample_count:06d}_len{length}.png"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                all_metadata['samples'].append(metadata)
                sample_count += 1
        
        # Add dataset info
        dataset_info = {
            'total_samples': sample_count,
            'samples_per_length': samples_per_length,
            'sequence_lengths': list(range(1, self.max_sequence_length + 1)),
            'image_size': self.image_size,
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generated_by': 'SyntheticDataGenerator v1.0 (balanced)'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'balanced' / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nBalanced dataset generated successfully!")
        print(f"Total samples: {sample_count}")
        print(f"Samples per length: {samples_per_length}")
        
        return all_metadata
    
    def preview_samples(self, num_samples: int = 10) -> List[Tuple[Image.Image, str]]:
        """
        Generate preview samples for visual inspection.
        
        Args:
            num_samples: Number of preview samples
            
        Returns:
            List of (image, label) tuples
        """
        samples = []
        
        for _ in range(num_samples):
            image, metadata = self.generate_single_image()
            samples.append((image, metadata['label']))
        
        return samples 