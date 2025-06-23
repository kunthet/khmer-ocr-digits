"""
Model factory and utilities for creating OCR models.

Provides factory methods for creating models from configuration files
and utility functions for model management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import yaml
import os

from .ocr_model import KhmerDigitsOCR
from .backbone import create_backbone
from .encoder import create_encoder
from .decoder import create_decoder


class ModelFactory:
    """
    Factory class for creating OCR models.
    
    Supports creating models from configuration files, presets,
    and custom parameters.
    """
    
    # Predefined model configurations
    MODEL_PRESETS = {
        'small': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 256,
            'encoder_hidden_size': 128,
            'decoder_hidden_size': 128,
            'attention_size': 128,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dropout': 0.1
        },
        'medium': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm', 
            'decoder_type': 'attention',
            'feature_size': 512,
            'encoder_hidden_size': 256,
            'decoder_hidden_size': 256,
            'attention_size': 256,
            'num_encoder_layers': 2,
            'num_decoder_layers': 1,
            'dropout': 0.1
        },
        'large': {
            'cnn_type': 'efficientnet-b0',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention', 
            'feature_size': 512,
            'encoder_hidden_size': 512,
            'decoder_hidden_size': 512,
            'attention_size': 512,
            'num_encoder_layers': 3,
            'num_decoder_layers': 2,
            'dropout': 0.1
        },
        'ctc_small': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'ctc',
            'feature_size': 256,
            'encoder_hidden_size': 128,
            'num_encoder_layers': 1,
            'dropout': 0.1
        },
        'ctc_medium': {
            'cnn_type': 'resnet18', 
            'encoder_type': 'bilstm',
            'decoder_type': 'ctc',
            'feature_size': 512,
            'encoder_hidden_size': 256,
            'num_encoder_layers': 2,
            'dropout': 0.1
        }
    }
    
    @classmethod
    def create_model(cls,
                    config: Optional[Union[str, Dict[str, Any]]] = None,
                    preset: Optional[str] = None,
                    **kwargs) -> KhmerDigitsOCR:
        """
        Create OCR model from configuration or preset.
        
        Args:
            config: Configuration dict or path to config file
            preset: Name of predefined model preset
            **kwargs: Additional model parameters to override
            
        Returns:
            KhmerDigitsOCR model instance
            
        Raises:
            ValueError: If both config and preset are provided or neither is provided
        """
        if config is not None and preset is not None:
            raise ValueError("Cannot specify both config and preset")
        
        if config is None and preset is None:
            # Use default medium preset
            preset = 'medium'
        
        # Get base parameters
        if preset is not None:
            if preset not in cls.MODEL_PRESETS:
                raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.MODEL_PRESETS.keys())}")
            model_params = cls.MODEL_PRESETS[preset].copy()
        else:
            # Load from config
            if isinstance(config, str):
                model_params = cls._load_config_file(config)
            else:
                model_params = cls._extract_model_params(config)
        
        # Override with kwargs
        model_params.update(kwargs)
        
        # Create model
        return KhmerDigitsOCR(**model_params)
    
    @classmethod
    def _load_config_file(cls, config_path: str) -> Dict[str, Any]:
        """Load model parameters from configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls._extract_model_params(config)
    
    @classmethod  
    def _extract_model_params(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model parameters from configuration dictionary."""
        model_config = config.get('model', {})
        
        # Map config to model parameters
        model_params = {
            'vocab_size': model_config.get('characters', {}).get('total_classes', 13),
            'max_sequence_length': model_config.get('characters', {}).get('max_sequence_length', 8),
            'cnn_type': model_config.get('cnn', {}).get('type', 'resnet18'),
            'encoder_type': model_config.get('rnn', {}).get('encoder', {}).get('type', 'bilstm'),
            'decoder_type': 'attention',  # Default to attention decoder
            'feature_size': model_config.get('cnn', {}).get('feature_size', 512),
            'encoder_hidden_size': model_config.get('rnn', {}).get('encoder', {}).get('hidden_size', 256),
            'decoder_hidden_size': model_config.get('rnn', {}).get('decoder', {}).get('hidden_size', 256),
            'attention_size': model_config.get('rnn', {}).get('attention', {}).get('hidden_size', 256),
            'num_encoder_layers': model_config.get('rnn', {}).get('encoder', {}).get('num_layers', 2),
            'num_decoder_layers': model_config.get('rnn', {}).get('decoder', {}).get('num_layers', 1),
            'dropout': model_config.get('rnn', {}).get('encoder', {}).get('dropout', 0.1),
            'pretrained_cnn': model_config.get('cnn', {}).get('pretrained', True)
        }
        
        return model_params
    
    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        List available model presets.
        
        Returns:
            Dictionary of preset names and their configurations
        """
        return cls.MODEL_PRESETS.copy()
    
    @classmethod
    def get_preset_info(cls, preset: str) -> Dict[str, Any]:
        """
        Get information about a specific preset.
        
        Args:
            preset: Name of the preset
            
        Returns:
            Preset configuration and estimated parameters
        """
        if preset not in cls.MODEL_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        
        config = cls.MODEL_PRESETS[preset].copy()
        
        # Estimate parameters (rough approximation)
        if config['cnn_type'] == 'resnet18':
            cnn_params = 11_000_000
        else:  # efficientnet-b0
            cnn_params = 5_000_000
        
        encoder_params = config['encoder_hidden_size'] * config['feature_size'] * 4 * config['num_encoder_layers']
        
        if config['decoder_type'] == 'attention':
            decoder_params = config['decoder_hidden_size'] * config['encoder_hidden_size'] * 4
        else:  # CTC
            decoder_params = config['encoder_hidden_size'] * 13  # vocab_size
        
        total_params = cnn_params + encoder_params + decoder_params
        
        info = {
            'configuration': config,
            'estimated_parameters': {
                'cnn': cnn_params,
                'encoder': encoder_params, 
                'decoder': decoder_params,
                'total': total_params
            },
            'estimated_size_mb': total_params * 4 / (1024 * 1024)
        }
        
        return info


def create_model(config_path: Optional[str] = None,
                preset: Optional[str] = None,
                **kwargs) -> KhmerDigitsOCR:
    """
    Convenience function to create OCR model.
    
    Args:
        config_path: Path to configuration file
        preset: Name of predefined preset
        **kwargs: Additional model parameters
        
    Returns:
        KhmerDigitsOCR model instance
    """
    return ModelFactory.create_model(config_path, preset, **kwargs)


def load_model(checkpoint_path: str,
              map_location: Optional[str] = None) -> KhmerDigitsOCR:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        map_location: Device to load model on
        
    Returns:
        Loaded KhmerDigitsOCR model
    """
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Extract model configuration
    if 'model_config' in checkpoint:
        model = KhmerDigitsOCR.from_config(checkpoint['model_config'])
    else:
        # Fallback to default configuration
        model = create_model(preset='medium')
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_model(model: KhmerDigitsOCR,
              checkpoint_path: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              epoch: Optional[int] = None,
              loss: Optional[float] = None,
              metrics: Optional[Dict[str, float]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: KhmerDigitsOCR model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        metrics: Current metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path) 