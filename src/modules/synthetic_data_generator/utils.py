"""
Utility functions for synthetic data generation.
"""

import os
import unicodedata
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import yaml
import numpy as np
from PIL import ImageFont


def normalize_khmer_text(text: str) -> str:
    """
    Normalize Khmer text using Unicode NFC normalization.
    
    Args:
        text: Input Khmer text
        
    Returns:
        Normalized Khmer text
    """
    return unicodedata.normalize('NFC', text)


def load_khmer_fonts(fonts_dir: str) -> Dict[str, str]:
    """
    Load all Khmer fonts from the fonts directory.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to font file paths
    """
    fonts = {}
    fonts_path = Path(fonts_dir)
    
    if not fonts_path.exists():
        raise FileNotFoundError(f"Fonts directory not found: {fonts_dir}")
    
    for font_file in fonts_path.glob("*.ttf"):
        font_name = font_file.stem
        fonts[font_name] = str(font_file)
    
    if not fonts:
        raise ValueError(f"No TTF fonts found in {fonts_dir}")
    
    return fonts


def get_khmer_digits() -> List[str]:
    """
    Get the list of Khmer digits.
    
    Returns:
        List of Khmer digit characters
    """
    return ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]


def get_special_tokens() -> List[str]:
    """
    Get the list of special tokens used in the model.
    
    Returns:
        List of special token strings
    """
    return ["<EOS>", "<PAD>", "<BLANK>"]


def create_character_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character to index and index to character mappings.
    
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    chars = get_khmer_digits() + get_special_tokens()
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return char_to_idx, idx_to_char


def generate_digit_sequence(min_length: int = 1, max_length: int = 8) -> str:
    """
    Generate a random sequence of Khmer digits.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        Random Khmer digit sequence
    """
    digits = get_khmer_digits()
    length = np.random.randint(min_length, max_length + 1)
    sequence = ''.join(np.random.choice(digits, size=length))
    return normalize_khmer_text(sequence)


def test_font_rendering(font_path: str, test_text: str = "០១២៣៤") -> bool:
    """
    Test if a font can properly render Khmer digits.
    
    Args:
        font_path: Path to font file
        test_text: Test text to render
        
    Returns:
        True if font renders properly, False otherwise
    """
    try:
        font = ImageFont.truetype(font_path, size=48)
        # Try to get text bounding box - this will fail if characters are not supported
        bbox = font.getbbox(test_text)
        return bbox[2] > bbox[0] and bbox[3] > bbox[1]  # width > 0 and height > 0
    except Exception:
        return False


def validate_font_collection(fonts_dir: str) -> Dict[str, bool]:
    """
    Validate all fonts in the collection for Khmer digit support.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to validation status
    """
    fonts = load_khmer_fonts(fonts_dir)
    validation_results = {}
    
    for font_name, font_path in fonts.items():
        validation_results[font_name] = test_font_rendering(font_path)
    
    return validation_results


def validate_dataset(dataset_path: str, expected_size: int) -> Dict[str, any]:
    """
    Validate a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        expected_size: Expected number of samples
        
    Returns:
        Dictionary with validation results
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        return {"valid": False, "error": "Dataset directory does not exist"}
    
    # Count image files
    image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
    
    # Check for metadata file
    metadata_file = dataset_dir / "metadata.yaml"
    has_metadata = metadata_file.exists()
    
    # Load and validate metadata if it exists
    metadata = None
    if has_metadata:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            return {"valid": False, "error": f"Failed to load metadata: {e}"}
    
    return {
        "valid": True,
        "num_images": len(image_files),
        "expected_size": expected_size,
        "size_match": len(image_files) == expected_size,
        "has_metadata": has_metadata,
        "metadata": metadata
    }


def calculate_dataset_statistics(dataset_path: str) -> Dict[str, any]:
    """
    Calculate statistics for a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    metadata_file = Path(dataset_path) / "metadata.yaml"
    
    if not metadata_file.exists():
        return {"error": "No metadata file found"}
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        return {"error": f"Failed to load metadata: {e}"}
    
    # Calculate statistics by combining train and val samples
    all_samples = []
    if 'train' in metadata and 'samples' in metadata['train']:
        all_samples.extend(metadata['train']['samples'])
    if 'val' in metadata and 'samples' in metadata['val']:
        all_samples.extend(metadata['val']['samples'])
    
    # If no train/val structure, try direct samples
    if not all_samples and 'samples' in metadata:
        all_samples = metadata['samples']
    
    sequence_lengths = [len(item['label']) for item in all_samples]
    fonts_used = [item['font'] for item in all_samples]
    
    stats = {
        "total_samples": len(all_samples),
        "sequence_length_distribution": {
            "min": min(sequence_lengths) if sequence_lengths else 0,
            "max": max(sequence_lengths) if sequence_lengths else 0,
            "mean": np.mean(sequence_lengths) if sequence_lengths else 0,
            "std": np.std(sequence_lengths) if sequence_lengths else 0
        },
        "font_distribution": {font: fonts_used.count(font) for font in set(fonts_used)},
        "character_frequency": {}
    }
    
    # Calculate character frequencies
    all_chars = ''.join([item['label'] for item in all_samples])
    for char in get_khmer_digits():
        stats["character_frequency"][char] = all_chars.count(char)
    
    return stats 