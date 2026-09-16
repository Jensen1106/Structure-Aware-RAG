"""LLM client wrapper (supports DashScope / OpenAI)"""

from typing import Optional
import dashscope
from dashscope import Generation
from src.config import LLM_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client"""

    def __init__(
        self,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.name = name or LLM_CONFIG["name"]
        self.api_key = api_key or LLM_CONFIG["api_key"]
        self.temperature = temperature if temperature is not None else LLM_CONFIG["temperature"]
        self.max_tokens = max_tokens or LLM_CONFIG["max_tokens"]

        if not self.api_key:
            logger.warning("API key not configured, LLM calls will fail")

    def generate(self, prompt: str) -> str:
        """Generate text"""
        try:
            response = Generation.call(
                model=self.name,
                api_key=self.api_key,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.status_code == 200:
                # DashScope format
                if hasattr(response.output, "text") and response.output.text:
                    return response.output.text
                # OpenAI-compatible format
                elif response.output and hasattr(response.output, "choices"):
                    return response.output.choices[0].message.content
                else:
                    logger.error(f"Unexpected response format: {response.output}")
                    return "[LLM response format error]"
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.message}")
                return f"[LLM API error: {response.status_code}]"

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM call failed: {e}]"

    def generate_with_messages(self, messages: list[dict]) -> str:
        """Generate using message list (supports multi-turn conversation)"""
        try:
            response = Generation.call(
                model=self.name,
                api_key=self.api_key,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.status_code == 200:
                if hasattr(response.output, "text") and response.output.text:
                    return response.output.text
                elif response.output and hasattr(response.output, "choices"):
                    return response.output.choices[0].message.content
                else:
                    return "[LLM response format error]"
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.message}")
                return f"[LLM API error: {response.status_code}]"

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM call failed: {e}]"


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def set_llm_client(client: LLMClient):
    """Set global LLM client"""
    global _llm_client
    _llm_client = client
