"""
Training Infrastructure for Khmer Digits OCR

This module provides comprehensive training infrastructure including:
- Base trainer classes with common functionality
- Specialized OCR trainer for sequence-to-sequence tasks
- Loss functions optimized for OCR tasks
- Evaluation metrics (character accuracy, sequence accuracy, edit distance)
- TensorBoard logging and model checkpointing
- Learning rate scheduling and early stopping
"""

from .base_trainer import BaseTrainer
from .ocr_trainer import OCRTrainer
from .losses import OCRLoss, CTCLoss, CrossEntropyLoss
from .metrics import OCRMetrics, calculate_character_accuracy, calculate_sequence_accuracy
from .utils import (
    TrainingConfig,
    CheckpointManager,
    EarlyStopping,
    setup_training_environment,
    save_training_config
)

__all__ = [
    # Trainers
    'BaseTrainer',
    'OCRTrainer',
    
    # Loss functions
    'OCRLoss',
    'CTCLoss', 
    'CrossEntropyLoss',
    
    # Metrics
    'OCRMetrics',
    'calculate_character_accuracy',
    'calculate_sequence_accuracy',
    
    # Utilities
    'TrainingConfig',
    'CheckpointManager',
    'EarlyStopping',
    'setup_training_environment',
    'save_training_config'
]

__version__ = "1.0.0" 