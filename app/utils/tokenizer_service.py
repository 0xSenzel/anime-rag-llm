import logging
from typing import Optional
import tiktoken
from tiktoken import Encoding

logger = logging.getLogger(__name__)

class TokenizerService:
    def __init__(self, tokenizer_encoding: str = "cl100k_base"):
        self.tokenizer: Optional[Encoding] = self._load_tokenizer(tokenizer_encoding) # <--- Also type hint self.tokenizer
        if not self.tokenizer:
            logger.warning(
                f"Tokenizer with encoding '{tokenizer_encoding}' not available. "
                "Token counting features will return 0."
            )

    def _load_tokenizer(self, encoding_name: str) -> Optional[Encoding]:
        """Loads the tiktoken tokenizer."""
        # if tiktoken: # This check is redundant if 'from tiktoken import Encoding' succeeds
        try:
            tokenizer = tiktoken.get_encoding(encoding_name)
            logger.info(f"Successfully loaded tiktoken tokenizer with encoding: {encoding_name}")
            return tokenizer
        except ImportError: # More specific exception for tiktoken not being installed
            logger.warning("tiktoken library not found. Install with `pip install tiktoken` for token counting.")
            return None
        except Exception as e: # Catch other potential errors from get_encoding
            logger.error(f"Failed to load tiktoken tokenizer encoding '{encoding_name}': {e}", exc_info=True)
            return None

    def count_tokens(self, text: str) -> int:
        """Estimates the number of tokens in a string using the loaded tokenizer."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error encoding text for token count: {e}", exc_info=True)
                return 0
        return 0