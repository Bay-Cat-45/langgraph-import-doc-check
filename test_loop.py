"""API を使わずにグラフの動きを確かめるテスト。

使い方: .venv/bin/python test_loop.py
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from graph import Extracted, Item, Review, Route, build_graph, find_issues

SAMPLE = Extracted(
    invoice_no="INV-1", date="2026-09-12", shipper="S", consignee="C", origin="China",
    incoterms="FOB", currency="USD", stated_total=16940.0,
    items=[
        Item(description="knife", hs_code="8211.91", qty=500, unit_price=12.0, amount=6000.0),
        Item(description="pan", hs_code="7323.93", qty=800, unit_price=9.5, amount=7800.0),
        Item(description="container", hs_code=None, qty=600, unit_price=3.0, amount=1800.0),
    ],
)


class FakeFast:
    """Haiku の代わり。スキーマごとに決めた答えを返す。"""

    def __init__(self, doc_type: str, verdicts: list[bool]):
        self.answers = {
            Route: iter([Route(doc_type=doc_type, reason="test")]),
            Extracted: iter([SAMPLE]),
            Review: iter(Review(approved=ok, feedback="" if ok else "不備の記載が不足") for ok in verdicts),
        }
        self.schema = None

    def with_structured_output(self, schema, include_raw=False):
        self.schema = schema
        return self

    def invoke(self, _prompt):
        usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        return {"raw": AIMessage(content="", usage_metadata=usage),
                "parsed": next(self.answers[self.schema]), "parsing_error": None}


def run(verdicts: list[bool], doc_type: str = "commercial_invoice") -> dict:
    smart = FakeListChatModel(responses=["regulation memo"] + [f"report v{i}" for i in range(1, 5)])
    app = build_graph(FakeFast(doc_type, verdicts), smart)
    return app.invoke({"document": "dummy"})


def test_find_issues():
    issues = find_issues(SAMPLE)
    assert len(issues) == 3, issues
    assert "800 × 9.50 = 7,600.00" in issues[0]
    assert "HSコードがない" in issues[1]
    assert "合計が合わない" in issues[2]


def test_router_skips_non_invoice():
    r = run([], doc_type="packing_list")
    assert "対象外" in r["report"] and "extracted" not in r
    assert [c["node"] for c in r["costs"]] == ["router"]


def test_approved_first_time():
    r = run([True])
    assert r["approved"] and r["revisions"] == 1 and r["report"] == "report v1"
    assert len(r["issues"]) == 3 and r["regulation"] == "regulation memo"


def test_approved_after_one_revision():
    r = run([False, True])
    assert r["approved"] and r["revisions"] == 2 and r["report"] == "report v2"


def test_stops_at_max_revisions():
    r = run([False, False, False, False])
    assert not r["approved"] and r["revisions"] == 3 and len(r["history"]) == 3


def test_costs_recorded_per_call():
    r = run([False, True])
    nodes = sorted(c["node"] for c in r["costs"])
    assert nodes == sorted(["router", "extractor", "regulation", "writer", "reviewer", "writer", "reviewer"])


if __name__ == "__main__":
    for t in (test_find_issues, test_router_skips_non_invoice, test_approved_first_time,
              test_approved_after_one_revision, test_stops_at_max_revisions, test_costs_recorded_per_call):
        t()
        print(f"PASS {t.__name__}")
