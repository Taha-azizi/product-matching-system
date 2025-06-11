from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # ... existing config ...
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict:
        """Load all prompt templates from the prompts folder in root directory."""
        prompts = {}
        # Use root directory prompts folder
        prompt_dir = Path(__file__).parent / "prompts"
        
        try:
            prompts['match_0shot'] = (prompt_dir / "prompt_0shot.txt").read_text(encoding='utf-8')
            prompts['match_fewshot'] = (prompt_dir / "prompt_fewshot.txt").read_text(encoding='utf-8')
            prompts['size_check'] = (prompt_dir / "prompt_size.txt").read_text(encoding='utf-8')
            
            logger.info(f"Successfully loaded {len(prompts)} prompt templates from {prompt_dir}")
            
        except FileNotFoundError as e:
            logger.error(f"Prompt file not found in {prompt_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise
            
        return prompts
    
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from file and initialize prompts."""
        # Your existing config loading logic
        config = cls()
        # ... load other config parameters ...
        return config