from logging import logger

class Config:
    def __init__(self):
        # ... existing config ...
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict:
        """Load all prompt templates from files."""
        prompts = {}
        prompt_dir = Path("prompts")
        
        try:
            prompts['match_0shot'] = (prompt_dir / "prompt_0shot.txt").read_text()
            prompts['match_fewshot'] = (prompt_dir / "prompt_fewshot.txt").read_text()
            prompts['size_check'] = (prompt_dir / "prompt_size.txt").read_text()
        except FileNotFoundError as e:
            logger.error(f"Prompt file not found: {e}")
            raise
            
        return prompts