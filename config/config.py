"""
Configuration Management
========================

Handles all configuration parameters for the product matching system.
"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the product matching system."""
    
    # Vector similarity parameters
    embedding_model: str = 'all-MiniLM-L6-v2'
    vector_similarity_threshold: float = 0.5
    top_k_matches: int = 3
    
    # Fuzzy matching parameters
    fuzzy_threshold: int = 85
    
    # LLM parameters
    llm_model: str = 'gemma3:27b'
    llm_temperature: float = 0.0
    
    # File paths
    prompt_dir: str = 'prompts'
    results_dir: str = 'results'
    
    # Processing parameters
    batch_size: int = 10
    max_retries: int = 3
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            Config object with loaded parameters
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Return default config if file doesn't exist
            return cls()
            
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'embedding_model': self.embedding_model,
            'vector_similarity_threshold': self.vector_similarity_threshold,
            'top_k_matches': self.top_k_matches,
            'fuzzy_threshold': self.fuzzy_threshold,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'prompt_dir': self.prompt_dir,
            'results_dir': self.results_dir,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries
        }