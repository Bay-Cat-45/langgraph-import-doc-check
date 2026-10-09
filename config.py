"""モデル名と API キーの確認。

API キーは環境変数 ANTHROPIC_API_KEY から読む（ファイルには置かない）。
"""
import os

SMART_MODEL = "claude-sonnet-5-5"          # 規制チェック・レポート作成
FAST_MODEL = "claude-haiku-4-5-20251001"   # 振り分け・抽出・レビュー
MAX_REVISIONS = 3


def load_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("環境変数 ANTHROPIC_API_KEY を設定してから実行してください")
