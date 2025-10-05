"""Token estimation utilities with tiktoken support."""
from typing import Dict, List, Optional
import tiktoken
from app.constants import (
    CHARS_PER_TOKEN,
    CHAT_FORMAT_OVERHEAD,
    BASE_CHAT_TOKENS,
    ROLE_FORMATTING_TOKENS,
    COMPLETION_PREP_TOKENS,
    DEFAULT_OUTPUT_TOKENS,
    MODEL_TOKEN_RATIOS,
)


class TokenEstimator:
    """Estimates token usage for LLM requests."""

    def __init__(self):
        """Initialize token estimator with tiktoken support."""
        self._encodings: Dict[str, tiktoken.Encoding] = {}

    def _get_encoding(self, model: str) -> Optional[tiktoken.Encoding]:
        """Get or create tiktoken encoding for model."""
        if model not in self._encodings:
            try:
                # Try to get encoding for specific model
                self._encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models (GPT-4, GPT-3.5-turbo)
                try:
                    self._encodings[model] = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    # If tiktoken fails completely, return None for fallback
                    return None
        return self._encodings.get(model)

    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate token count for text using tiktoken if available,
        fallback to heuristic.
        """
        if not text:
            return 0

        # Try tiktoken first for accuracy
        encoding = self._get_encoding(model)
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass  # Fall back to heuristic

        # Heuristic fallback
        ratio = MODEL_TOKEN_RATIOS.get(model, CHARS_PER_TOKEN)
        estimated = max(1, int(len(text) / ratio))

        # Add overhead for chat format
        if "chat" in model or "gpt" in model:
            estimated = int(estimated * CHAT_FORMAT_OVERHEAD)

        return estimated

    def estimate_chat_tokens(
        self,
        messages: List[Dict],
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Estimate tokens for a chat conversation.

        Returns:
            Dict with 'input_tokens' and 'output_tokens' estimates.
        """
        input_tokens = BASE_CHAT_TOKENS

        # Estimate input tokens
        for message in messages:
            content = message.get("content", "")
            input_tokens += ROLE_FORMATTING_TOKENS
            input_tokens += self.estimate_tokens(content, model)

        # Additional overhead for completion
        input_tokens += COMPLETION_PREP_TOKENS

        # Estimate output tokens
        output_tokens = max_tokens if max_tokens else DEFAULT_OUTPUT_TOKENS

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def estimate_messages_tokens(
        self,
        messages: List[Dict],
        model: str = "gpt-4o-mini"
    ) -> int:
        """Estimate total tokens for messages (input only)."""
        estimates = self.estimate_chat_tokens(messages, model)
        return estimates["input_tokens"]


# Global singleton
token_estimator = TokenEstimator()
