"""グラフ構造を Mermaid 形式で graph.mmd に書き出す（API・ネット接続なし）。

使い方: .venv/bin/python draw_graph.py
"""
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from graph import build_graph

dummy = FakeListChatModel(responses=[""])
mermaid = build_graph(dummy, dummy).get_graph().draw_mermaid()
Path(__file__).with_name("graph.mmd").write_text(mermaid, encoding="utf-8")
print(mermaid)
