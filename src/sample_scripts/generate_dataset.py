#!/usr/bin/env python3
"""
Script to generate synthetic training data for Khmer digits OCR.

This script demonstrates the usage of the SyntheticDataGenerator
and creates the initial training dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator import SyntheticDataGenerator
from modules.synthetic_data_generator.utils import validate_font_collection, calculate_dataset_statistics


def main():
    """Main function to generate synthetic dataset."""
    parser = argparse.ArgumentParser(description='Generate synthetic Khmer digits dataset')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='Path to model configuration file')
    parser.add_argument('--fonts-dir', type=str, default='src/fonts',
                      help='Directory containing Khmer fonts')
    parser.add_argument('--output-dir', type=str, default='generated_data',
                      help='Output directory for generated dataset')
    parser.add_argument('--num-samples', type=int, default=1000,
                      help='Number of samples to generate')
    parser.add_argument('--train-split', type=float, default=0.8,
                      help='Fraction of samples for training')
    parser.add_argument('--preview-only', action='store_true',
                      help='Only generate preview samples without saving')
    parser.add_argument('--validate-fonts', action='store_true',
                      help='Validate font collection before generation')
    
    args = parser.parse_args()
    
    # Validate paths
    config_path = Path(args.config)
    fonts_dir = Path(args.fonts_dir)
    output_dir = Path(args.output_dir)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    if not fonts_dir.exists():
        print(f"Error: Fonts directory not found: {fonts_dir}")
        return 1
    
    print("=== Khmer Digits Synthetic Data Generator ===")
    print(f"Configuration: {config_path}")
    print(f"Fonts directory: {fonts_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Train/validation split: {args.train_split:.1%}/{1-args.train_split:.1%}")
    print()
    
    # Validate fonts if requested
    if args.validate_fonts:
        print("Validating font collection...")
        validation_results = validate_font_collection(str(fonts_dir))
        
        print("Font validation results:")
        for font_name, is_valid in validation_results.items():
            status = "✓" if is_valid else "✗"
            print(f"  {status} {font_name}")
        
        working_fonts = [name for name, valid in validation_results.items() if valid]
        print(f"\nWorking fonts: {len(working_fonts)}/{len(validation_results)}")
        
        if not working_fonts:
            print("Error: No working fonts found!")
            return 1
        print()
    
    try:
        # Initialize generator
        print("Initializing synthetic data generator...")
        generator = SyntheticDataGenerator(
            config_path=str(config_path),
            fonts_dir=str(fonts_dir),
            output_dir=str(output_dir)
        )
        print("Generator initialized successfully!")
        print()
        
        if args.preview_only:
            # Generate preview samples
            print("Generating preview samples...")
            samples = generator.preview_samples(num_samples=10)
            
            print("Preview samples generated:")
            for i, (image, label) in enumerate(samples):
                print(f"  Sample {i+1}: '{label}' ({len(label)} digit(s))")
            
            # Save preview samples
            preview_dir = output_dir / 'preview'
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (image, label) in enumerate(samples):
                filename = f"preview_{i:02d}_{label}.png"
                image.save(preview_dir / filename)
            
            print(f"\nPreview samples saved to: {preview_dir}")
        
        else:
            # Generate full dataset
            print("Generating synthetic dataset...")
            metadata = generator.generate_dataset(
                num_samples=args.num_samples,
                train_split=args.train_split,
                save_images=True,
                show_progress=True
            )
            
            # Calculate and display statistics
            print("\nCalculating dataset statistics...")
            stats = calculate_dataset_statistics(str(output_dir))
            
            if 'error' not in stats:
                print("Dataset Statistics:")
                print(f"  Total samples: {stats['total_samples']}")
                print(f"  Sequence length range: {stats['sequence_length_distribution']['min']}-{stats['sequence_length_distribution']['max']}")
                print(f"  Average sequence length: {stats['sequence_length_distribution']['mean']:.1f}")
                
                print("\n  Font distribution:")
                for font, count in stats['font_distribution'].items():
                    percentage = (count / stats['total_samples']) * 100
                    print(f"    {font}: {count} ({percentage:.1f}%)")
                
                print("\n  Character frequency:")
                for char, count in stats['character_frequency'].items():
                    print(f"    '{char}': {count}")
            else:
                print(f"Error calculating statistics: {stats['error']}")
        
        print("\n=== Generation completed successfully! ===")
        return 0
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main()) 