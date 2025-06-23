"""
Complete Khmer Digits OCR Model

Combines CNN backbone, RNN encoder, attention mechanism, and decoder
into a unified architecture for end-to-end Khmer digit sequence recognition.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
import yaml

from .backbone import CNNBackbone, create_backbone
from .encoder import RNNEncoder, create_encoder  
from .decoder import RNNDecoder, create_decoder
from .attention import BahdanauAttention


class KhmerDigitsOCR(nn.Module):
    """
    Complete Khmer Digits OCR Model.
    
    End-to-end architecture combining:
    - CNN backbone for feature extraction
    - RNN encoder for sequence modeling
    - Attention mechanism for alignment
    - RNN decoder for character generation
    """
    
    def __init__(self,
                 vocab_size: int = 13,
                 max_sequence_length: int = 8,
                 cnn_type: str = 'resnet18',
                 encoder_type: str = 'bilstm',
                 decoder_type: str = 'attention',
                 feature_size: int = 512,
                 encoder_hidden_size: int = 256,
                 decoder_hidden_size: int = 256,
                 attention_size: int = 256,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dropout: float = 0.1,
                 pretrained_cnn: bool = True):
        """
        Initialize Khmer Digits OCR model.
        
        Args:
            vocab_size: Size of character vocabulary (10 digits + 3 special tokens)
            max_sequence_length: Maximum sequence length
            cnn_type: Type of CNN backbone ('resnet18', 'efficientnet-b0')
            encoder_type: Type of encoder ('bilstm', 'conv')
            decoder_type: Type of decoder ('attention', 'ctc')
            feature_size: Size of CNN output features
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states
            attention_size: Size of attention mechanism
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            pretrained_cnn: Whether to use pretrained CNN weights
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.feature_size = feature_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        # Character mappings (Khmer digits + special tokens)
        self.char_to_idx = {
            '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
            '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
            '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # CNN Backbone
        self.backbone = create_backbone(
            backbone_type=cnn_type,
            feature_size=feature_size,
            pretrained=pretrained_cnn,
            dropout=dropout
        )
        
        # RNN Encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Decoder
        if decoder_type == 'attention':
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                num_layers=num_decoder_layers,
                dropout=dropout,
                attention_size=attention_size
            )
        else:  # CTC decoder
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                dropout=dropout
            )
        
        self.decoder_type = decoder_type
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # CNN backbone weights are handled by the backbone itself
        # Encoder and decoder weights are handled by their respective classes
        pass
    
    def forward(self,
                images: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None,
                allow_early_stopping: bool = False) -> torch.Tensor:
        """
        Forward pass of the OCR model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            target_sequences: Target sequences for training [batch_size, seq_len] (optional)
            sequence_lengths: Actual sequence lengths [batch_size] (optional)
            
        Returns:
            Character predictions [batch_size, seq_len, vocab_size]
        """
        # Extract CNN features
        cnn_features = self.backbone(images)  # [batch_size, cnn_seq_len, feature_size]
        
        # Encode features
        encoder_features, final_hidden = self.encoder(cnn_features)  # [batch_size, seq_len, encoder_hidden_size]
        
        # Decode to character sequences
        if self.decoder_type == 'attention':
            # Use the provided allow_early_stopping parameter
            predictions = self.decoder(
                encoder_features, 
                target_sequences, 
                self.max_sequence_length,
                allow_early_stopping
            )
        else:  # CTC decoder
            predictions = self.decoder(encoder_features)
        
        return predictions
    
    def predict(self,
                images: torch.Tensor,
                return_attention: bool = False) -> List[str]:
        """
        Predict character sequences from images.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            return_attention: Whether to return attention weights
            
        Returns:
            List of predicted text strings
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with early stopping enabled for actual inference
            predictions = self.forward(images, allow_early_stopping=True)  # [batch_size, seq_len, vocab_size]
            
            # Convert to character indices
            predicted_indices = torch.argmax(predictions, dim=-1)  # [batch_size, seq_len]
            
            # Decode to text
            texts = []
            for sequence in predicted_indices:
                text = self._decode_sequence(sequence)
                texts.append(text)
            
            return texts
    
    def _decode_sequence(self, indices: torch.Tensor) -> str:
        """
        Decode a sequence of character indices to text.
        
        Args:
            indices: Character indices [seq_len]
            
        Returns:
            Decoded text string
        """
        chars = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx_to_char:
                char = self.idx_to_char[idx_val]
                if char in ['<EOS>', '<PAD>', '<BLANK>']:
                    break
                chars.append(char)
        
        return ''.join(chars)
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode text string to character indices.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (uses model default if None)
            
        Returns:
            Character indices [seq_len]
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        # Convert characters to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Skip unknown characters
                continue
        
        # Add EOS token
        indices.append(self.char_to_idx['<EOS>'])
        
        # Pad to max length
        while len(indices) < max_length:
            indices.append(self.char_to_idx['<PAD>'])
        
        # Truncate if too long
        indices = indices[:max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        info = {
            'model_name': 'KhmerDigitsOCR',
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'feature_size': self.feature_size,
            'encoder_hidden_size': self.encoder_hidden_size,
            'decoder_hidden_size': self.decoder_hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': {
                'backbone': backbone_params,
                'encoder': encoder_params,
                'decoder': decoder_params
            },
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'character_mappings': self.char_to_idx
        }
        
        return info
    
    @classmethod
    def from_config(cls, config_path: str) -> 'KhmerDigitsOCR':
        """
        Create model from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            KhmerDigitsOCR model instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        
        # Extract model parameters
        model_params = {
            'vocab_size': model_config.get('characters', {}).get('total_classes', 13),
            'max_sequence_length': model_config.get('characters', {}).get('max_sequence_length', 8),
            'cnn_type': model_config.get('cnn', {}).get('type', 'resnet18'),
            'feature_size': model_config.get('cnn', {}).get('feature_size', 512),
            'encoder_hidden_size': model_config.get('rnn', {}).get('encoder', {}).get('hidden_size', 256),
            'decoder_hidden_size': model_config.get('rnn', {}).get('decoder', {}).get('hidden_size', 256),
            'attention_size': model_config.get('rnn', {}).get('attention', {}).get('hidden_size', 256),
            'num_encoder_layers': model_config.get('rnn', {}).get('encoder', {}).get('num_layers', 2),
            'num_decoder_layers': model_config.get('rnn', {}).get('decoder', {}).get('num_layers', 1),
            'dropout': model_config.get('rnn', {}).get('encoder', {}).get('dropout', 0.1),
            'pretrained_cnn': model_config.get('cnn', {}).get('pretrained', True)
        }
        
        return cls(**model_params)
    
    def save_config(self, config_path: str):
        """
        Save model configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config = {
            'model': {
                'name': 'khmer_digits_ocr',
                'architecture': 'cnn_rnn_attention',
                'vocab_size': self.vocab_size,
                'max_sequence_length': self.max_sequence_length,
                'feature_size': self.feature_size,
                'encoder_hidden_size': self.encoder_hidden_size,
                'decoder_hidden_size': self.decoder_hidden_size,
                'character_mappings': self.char_to_idx
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True) 