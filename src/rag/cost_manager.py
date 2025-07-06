"""Cost management utilities for RAG system."""

import sys
from typing import Dict, Tuple

from .token_counter import TokenCounter


class CostManager:
    """Manages cost estimation and limits for RAG operations."""

    def __init__(self, config: dict):
        """Initialize CostManager.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.embedding_counter = TokenCounter("text-embedding-3-small")
        self.completion_counter = TokenCounter(
            config.get("completion_model", "gpt-3.5-turbo")
        )

        # Cost limits (USD)
        self.cost_limits = {
            "per_query": config.get("cost_limit_per_query", 0.10),  # $0.10 per query
            "session_total": config.get(
                "cost_limit_session", 1.00
            ),  # $1.00 per session
            "warning_threshold": config.get(
                "cost_warning_threshold", 0.05
            ),  # $0.05 warning
        }

        # Session tracking
        self.session_cost = 0.0
        self.query_count = 0

    def estimate_query_cost(
        self, query: str, chunks: list, context_length: int = None
    ) -> Dict[str, float]:
        """Estimate cost for a query before execution.

        Args:
            query (str): User query
            chunks (list): Retrieved chunks
            context_length (int): Optional context length override

        Returns:
            dict: Cost breakdown
        """
        # Estimate embedding cost (for query)
        query_tokens = self.embedding_counter.count_tokens(query)
        embedding_cost = self.embedding_counter.estimate_cost(query_tokens)

        # Estimate completion cost
        if context_length is None:
            # Estimate context from chunks
            context_text = ""
            for chunk in chunks:
                context_text += chunk.get("content", "")
            context_tokens = self.completion_counter.count_tokens(context_text + query)
        else:
            context_tokens = context_length

        # Estimate output tokens (rough estimate)
        estimated_output_tokens = min(
            context_tokens // 4, 500
        )  # Assume 1/4 of input or max 500

        input_cost = self.completion_counter.estimate_cost(
            context_tokens, is_output=False
        )
        output_cost = self.completion_counter.estimate_cost(
            estimated_output_tokens, is_output=True
        )
        completion_cost = input_cost + output_cost

        total_cost = embedding_cost + completion_cost

        return {
            "embedding_cost": embedding_cost,
            "completion_input_cost": input_cost,
            "completion_output_cost": output_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "estimated_tokens": {
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "estimated_output_tokens": estimated_output_tokens,
            },
        }

    def check_cost_limits(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if estimated cost exceeds limits.

        Args:
            estimated_cost (float): Estimated cost in USD

        Returns:
            tuple: (should_proceed, message)
        """
        messages = []

        # Check per-query limit
        if estimated_cost > self.cost_limits["per_query"]:
            messages.append(
                f"⚠️  クエリコストが上限を超えています: "
                f"${estimated_cost:.4f} > ${self.cost_limits['per_query']:.4f}"
            )

        # Check session total limit
        projected_session_cost = self.session_cost + estimated_cost
        if projected_session_cost > self.cost_limits["session_total"]:
            messages.append(
                f"⚠️  セッション合計コストが上限を超えます: "
                f"${projected_session_cost:.4f} > ${self.cost_limits['session_total']:.4f}"
            )

        # Check warning threshold
        elif estimated_cost > self.cost_limits["warning_threshold"]:
            messages.append(
                f"💡 コスト警告: ${estimated_cost:.4f} "
                f"(警告閾値: ${self.cost_limits['warning_threshold']:.4f})"
            )

        if messages:
            return False, "\n".join(messages)

        return True, ""

    def prompt_user_confirmation(
        self, estimated_cost: float, cost_breakdown: dict
    ) -> bool:
        """Prompt user for confirmation when cost exceeds limits.

        Args:
            estimated_cost (float): Estimated cost
            cost_breakdown (dict): Detailed cost breakdown

        Returns:
            bool: User's decision to proceed
        """
        print("\n" + "=" * 50)
        print("💰 コスト確認")
        print("=" * 50)
        print(f"予想コスト: ${estimated_cost:.4f}")
        print(f"セッション累計: ${self.session_cost:.4f}")
        print(f"予想セッション合計: ${self.session_cost + estimated_cost:.4f}")
        print()
        print("詳細内訳:")
        print(f"  - 埋め込み: ${cost_breakdown['embedding_cost']:.6f}")
        print(f"  - 補完(入力): ${cost_breakdown['completion_input_cost']:.6f}")
        print(f"  - 補完(出力): ${cost_breakdown['completion_output_cost']:.6f}")
        print()
        print("トークン情報:")
        tokens = cost_breakdown["estimated_tokens"]
        print(f"  - クエリトークン: {tokens['query_tokens']}")
        print(f"  - コンテキストトークン: {tokens['context_tokens']}")
        print(f"  - 予想出力トークン: {tokens['estimated_output_tokens']}")
        print("=" * 50)

        while True:
            response = input("続行しますか？ (y/n/q): ").strip().lower()
            if response in ["y", "yes", "はい"]:
                return True
            elif response in ["n", "no", "いいえ"]:
                return False
            elif response in ["q", "quit", "終了"]:
                print("セッションを終了します。")
                sys.exit(0)
            else:
                print("y(はい), n(いいえ), q(終了) のいずれかを入力してください。")

    def record_actual_cost(self, actual_cost: float):
        """Record actual cost after query execution.

        Args:
            actual_cost (float): Actual cost incurred
        """
        self.session_cost += actual_cost
        self.query_count += 1

    def get_session_summary(self) -> dict:
        """Get session cost summary.

        Returns:
            dict: Session summary
        """
        return {
            "total_cost": self.session_cost,
            "query_count": self.query_count,
            "average_cost_per_query": self.session_cost / max(self.query_count, 1),
            "remaining_budget": max(
                0, self.cost_limits["session_total"] - self.session_cost
            ),
        }

    def print_session_summary(self):
        """Print session cost summary."""
        summary = self.get_session_summary()
        print("\n" + "=" * 40)
        print("📊 セッションサマリー")
        print("=" * 40)
        print(f"合計コスト: ${summary['total_cost']:.4f}")
        print(f"クエリ数: {summary['query_count']}")
        print(f"平均コスト/クエリ: ${summary['average_cost_per_query']:.4f}")
        print(f"残り予算: ${summary['remaining_budget']:.4f}")
        print("=" * 40)
