"""ノードごとのトークン数と料金（USD）を記録する。"""
from langchain_core.messages import AIMessage

# USD / 100万トークン（入力, 出力）。Anthropic 公式料金 2026-09-25 時点
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def cost_entry(node: str, llm, message: AIMessage) -> dict:
    model = getattr(llm, "model", "fake")
    usage = message.usage_metadata or {}
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)  # thinking トークンも出力として課金される
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    usd = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    return {"node": node, "model": model, "in": tokens_in, "out": tokens_out, "usd": usd}


def format_costs(costs: list[dict]) -> str:
    lines = [f"{'node':<11}{'model':<28}{'in':>7}{'out':>7}{'USD':>10}"]
    for c in costs:
        lines.append(f"{c['node']:<11}{c['model']:<28}{c['in']:>7}{c['out']:>7}{c['usd']:>10.4f}")
    total = sum(c["usd"] for c in costs)
    lines.append(f"{'合計':<44}{sum(c['in'] for c in costs):>7}{sum(c['out'] for c in costs):>7}{total:>10.4f}")
    return "\n".join(lines)
