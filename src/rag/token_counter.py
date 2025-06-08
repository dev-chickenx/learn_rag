"""Token counting and cost estimation utilities."""

import tiktoken


class TokenCounter:
    # OpenAI APIの料金（1000トークンあたりのUSD）
    # 参考: https://openai.com/pricing
    COST_RATES = {
        "gpt-3.5-turbo": {
            "input": 0.0005,  # $0.0005 / 1K tokens
            "output": 0.0015,  # $0.0015 / 1K tokens
        },
        "gpt-4": {
            "input": 0.03,  # $0.03 / 1K tokens
            "output": 0.06,  # $0.06 / 1K tokens
        },
        "text-embedding-3-small": {
            "input": 0.00002,  # $0.00002 / 1K tokens
            "output": 0.00002,  # $0.00002 / 1K tokens
        },
    }

    def __init__(self, model_name: str):
        """Initialize TokenCounter.

        Args:
            model_name (str): Name of the model to use for counting tokens
        """
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.total_tokens = 0
        self.total_cost = 0.0

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text (str): Text to count tokens for

        Returns:
            int: Number of tokens
        """
        tokens = self.encoding.encode(text)
        return len(tokens)

    def estimate_cost(self, num_tokens: int, is_output: bool = False) -> float:
        """Estimate cost for the given number of tokens.

        OpenAIの料金体系は1000トークンあたりの価格で定義されているため、
        実際のコストを計算するには (トークン数 ÷ 1000) × (1Kトークンあたりの料金) となります。

        Args:
            num_tokens (int): Number of tokens
            is_output (bool): Whether these are output tokens

        Returns:
            float: Estimated cost in USD
        """
        token_type = "output" if is_output else "input"
        rate = self.COST_RATES[self.model_name][token_type]
        return (num_tokens / 1000) * rate

    def add_to_total(self, text: str, is_output: bool = False):
        """Add tokens and cost to running total.

        Args:
            text (str): Text to count tokens for
            is_output (bool): Whether these are output tokens
        """
        tokens = self.count_tokens(text)
        cost = self.estimate_cost(tokens, is_output)

        self.total_tokens += tokens
        self.total_cost += cost

    def get_total_stats(self) -> dict:
        """Get total token count and cost statistics.

        Returns:
            dict: Dictionary containing total tokens and cost
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }

    def reset_totals(self):
        """Reset running totals to zero."""
        self.total_tokens = 0
        self.total_cost = 0.0
