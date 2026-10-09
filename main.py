"""使い方: .venv/bin/python main.py samples/invoice_001.txt

結果は画面に出し、同じ内容を runs/run_NNN.txt にも保存する。
"""
import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic

from config import FAST_MODEL, SMART_MODEL, load_api_key
from costs import format_costs
from graph import build_graph

RUNS_DIR = Path(__file__).parent / "runs"


def render(result: dict) -> str:
    out = [f"書類判定: {result['doc_type']}"]
    if result.get("issues") is not None:
        out.append("\n===== プログラムで確認した不備 =====")
        out += [f"- {i}" for i in result["issues"]] or ["なし"]
    for h in result.get("history", []):
        mark = "OK" if h["approved"] else "NG"
        out.append(f"[round {h['round']}] {mark} {h['feedback']}")
    out += ["\n===== 最終レポート =====\n", result["report"]]
    if result.get("history") and not result["approved"]:
        out.append("\n※ 差し戻し上限に達したため、未承認のまま終了しました")
    out += ["\n===== コスト =====", format_costs(result["costs"])]
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    load_api_key()
    app = build_graph(
        fast_llm=ChatAnthropic(model=FAST_MODEL, max_tokens=4000),
        # Sonnet 5.5 は thinking が既定でオン。effort で深さを抑え、本文の出力枠を確保する
        smart_llm=ChatAnthropic(model=SMART_MODEL, max_tokens=16000, output_config={"effort": "medium"}),
    )
    text = render(app.invoke({"document": Path(sys.argv[1]).read_text(encoding="utf-8")}))
    print(text)

    RUNS_DIR.mkdir(exist_ok=True)
    run_file = RUNS_DIR / f"run_{len(list(RUNS_DIR.glob('run_*.txt'))) + 1:03d}.txt"
    run_file.write_text(text + "\n", encoding="utf-8")
    print(f"\n保存しました: {run_file}")


if __name__ == "__main__":
    main()
