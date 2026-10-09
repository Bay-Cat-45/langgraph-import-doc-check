"""輸入書類チェック：振り分け → 読取 → 計算/規制チェック（並列）→ レポート → レビュー（差し戻し）

    START → router ──対象外──→ END
              │
         インボイス
              ↓
          extractor ─┬→ calc_check ─┐
                     └→ regulation ─┴→ writer → reviewer ──OK──→ END
                                          ↑          │
                                          └─NG(≦3回)─┘

- 軽い判定・抽出・レビューは Haiku、考える仕事（規制・レポート）は Sonnet
- 計算チェックは LLM を使わず Python で確定させる（数字の誤りを AI に任せない）
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import MAX_REVISIONS
from costs import cost_entry


# ---------- 構造化出力のスキーマ ----------

class Route(BaseModel):
    doc_type: Literal["commercial_invoice", "packing_list", "other"] = Field(
        description="書類の種類。コマーシャルインボイスなら commercial_invoice"
    )
    reason: str = Field(description="判定理由を1文で")


class Item(BaseModel):
    description: str
    hs_code: str | None = Field(description="記載がなければ null")
    qty: float
    unit_price: float
    amount: float = Field(description="インボイスに記載された金額（再計算しない）")


class Extracted(BaseModel):
    invoice_no: str | None
    date: str | None
    shipper: str | None
    consignee: str | None
    origin: str | None = Field(description="原産国")
    incoterms: str | None
    currency: str | None
    items: list[Item]
    stated_total: float | None = Field(description="インボイスに記載された合計金額（再計算しない）")


class Review(BaseModel):
    approved: bool = Field(description="レポートが基準を満たしていれば true")
    feedback: str = Field(description="NGの場合は具体的な修正指示。OKなら空文字")


# ---------- State ----------

class State(TypedDict):
    document: str
    doc_type: str
    extracted: dict
    issues: list[str]       # calc_check がプログラムで確定させた不備
    regulation: str         # regulation の検討メモ
    report: str
    feedback: str
    approved: bool
    revisions: int          # writer を走らせた回数
    history: list[dict]
    costs: Annotated[list[dict], operator.add]  # 並列ノードからも追記できるよう reducer で連結


# ---------- プロンプト ----------

ROUTER_PROMPT = """次の書類の種類を判定してください。

--- 書類 ---
{document}
"""

EXTRACTOR_PROMPT = """次のコマーシャルインボイスから項目を抜き出してください。
書かれている値をそのまま写し、計算し直したり補ったりしないこと。記載がない項目は null にすること。

--- インボイス ---
{document}
"""

REGULATION_PROMPT = """あなたは日本の輸入通関に詳しい担当者です。
次の品目それぞれについて、日本へ輸入する際に関係しそうな法規（食品衛生法、電気用品安全法、銃刀法、各種表示法など）と、
用意すべき追加書類・手続きを簡潔に挙げてください。
品目名だけでは決められないものは「要確認」とし、何を確認すれば決まるかを書くこと。

原産国: {origin}
--- 品目 ---
{items}
"""

WRITER_PROMPT = """あなたは輸入書類チェックの担当者です。材料をもとに、荷主と通関担当が読む日本語のチェックレポートを作成してください。

構成:
1. 記載漏れ・不備（下の「プログラムで確認した不備」は計算で確定した事実なので、すべてそのまま載せること）
2. 品目ごとの関連法規と必要な追加書類
3. 通関前に荷主へ確認すべき事項（箇条書き）

推測で断定せず、確認が必要なものは「要確認」と書くこと。

--- インボイス原文 ---
{document}

--- プログラムで確認した不備 ---
{issues}

--- 規制チェックのメモ ---
{regulation}
"""

REVISION_SUFFIX = """
--- 前回のレポート ---
{report}

--- レビュー担当からの差し戻し ---
{feedback}

指摘をすべて反映した改訂版レポートを、全文で書き直してください。
"""

REVIEWER_PROMPT = """あなたは輸入書類チェックレポートのレビュー担当です。
元のインボイスとレポートを照らし合わせ、次の基準で合否を判定してください。

- 「プログラムで確認した不備」がすべてレポートに載っているか（これは計算で確定した事実）
- 品目ごとに関連法規と必要書類に触れているか
- 根拠のない断定をしていないか
- 荷主への確認事項が具体的か

判定は「通関実務で使えるか」で行い、完璧さは求めないこと。
基準を満たさない重大な欠陥がある場合だけ approved=false とし、feedback に修正指示を重要なものから最大5項目、各1〜2文で書くこと。
インボイスの情報だけでは確定できない事項を、レポートに確定させるよう求めてはいけない（「要確認」で正しい）。

--- インボイス ---
{document}

--- プログラムで確認した不備 ---
{issues}

--- レポート ---
{report}
"""

REVIEWER_FOLLOWUP = """
--- 前回あなたが出した差し戻し ---
{feedback}

今回は、上の指摘が解消されたかを中心に確認すること。
解消されていて、新たな重大な欠陥がなければ approved=true とすること。
軽微な改善点を理由に、新しい要求を追加して差し戻さないこと。
"""


# ---------- LLM を使わないチェック ----------

REQUIRED_FIELDS = {
    "invoice_no": "インボイス番号", "date": "日付", "shipper": "荷送人", "consignee": "荷受人",
    "origin": "原産国", "incoterms": "インコタームズ", "currency": "通貨",
}


def find_issues(ex: Extracted) -> list[str]:
    """記載漏れと計算の不一致を、計算で確定させる。"""
    issues = [f"{label}の記載がない" for key, label in REQUIRED_FIELDS.items() if not getattr(ex, key)]
    for no, item in enumerate(ex.items, 1):
        if not item.hs_code:
            issues.append(f"明細{no}（{item.description}）にHSコードがない")
        expected = round(item.qty * item.unit_price, 2)
        if abs(expected - item.amount) > 0.005:
            issues.append(
                f"明細{no}（{item.description}）の金額が合わない: "
                f"{item.qty:g} × {item.unit_price:,.2f} = {expected:,.2f} だが記載は {item.amount:,.2f}"
            )
    line_sum = round(sum(i.amount for i in ex.items), 2)
    if ex.stated_total is None:
        issues.append("合計金額の記載がない")
    elif abs(line_sum - ex.stated_total) > 0.005:
        issues.append(f"合計が合わない: 明細金額の和は {line_sum:,.2f} だが記載は {ex.stated_total:,.2f}")
    return issues


# ---------- グラフ ----------

def build_graph(fast_llm: BaseChatModel, smart_llm: BaseChatModel):
    """fast_llm: 振り分け・抽出・レビュー用（Haiku） / smart_llm: 規制チェック・レポート作成用（Sonnet）"""

    def structured(schema, prompt: str, node: str):
        out = fast_llm.with_structured_output(schema, include_raw=True).invoke(prompt)
        if out["parsing_error"]:
            raise RuntimeError(f"{node} の構造化出力に失敗しました: {out['parsing_error']}")
        return out["parsed"], cost_entry(node, fast_llm, out["raw"])

    def think(prompt: str, node: str):
        response = smart_llm.invoke(prompt)
        if response.response_metadata.get("stop_reason") == "max_tokens":
            raise RuntimeError(f"{node} の出力が max_tokens で途中終了しました")
        # thinking ブロックを除いた本文だけを使う
        return response.text, cost_entry(node, smart_llm, response)

    def router(state: State) -> dict:
        route, cost = structured(Route, ROUTER_PROMPT.format(document=state["document"]), "router")
        update = {"doc_type": route.doc_type, "costs": [cost]}
        if route.doc_type != "commercial_invoice":
            update["report"] = f"対象外の書類のためチェックしませんでした（判定: {route.doc_type} / {route.reason}）"
        return update

    def extractor(state: State) -> dict:
        ex, cost = structured(Extracted, EXTRACTOR_PROMPT.format(document=state["document"]), "extractor")
        return {"extracted": ex.model_dump(), "costs": [cost]}

    def calc_check(state: State) -> dict:
        return {"issues": find_issues(Extracted(**state["extracted"]))}

    def regulation(state: State) -> dict:
        ex = Extracted(**state["extracted"])
        items = "\n".join(f"- {i.description}（HS: {i.hs_code or '記載なし'}）" for i in ex.items)
        memo, cost = think(REGULATION_PROMPT.format(origin=ex.origin or "記載なし", items=items), "regulation")
        return {"regulation": memo, "costs": [cost]}

    def writer(state: State) -> dict:
        prompt = WRITER_PROMPT.format(
            document=state["document"],
            issues="\n".join(f"- {i}" for i in state["issues"]) or "なし",
            regulation=state["regulation"],
        )
        if state.get("feedback"):
            prompt += REVISION_SUFFIX.format(report=state["report"], feedback=state["feedback"])
        report, cost = think(prompt, "writer")
        return {"report": report, "revisions": state.get("revisions", 0) + 1, "costs": [cost]}

    def reviewer(state: State) -> dict:
        prompt = REVIEWER_PROMPT.format(
            document=state["document"],
            issues="\n".join(f"- {i}" for i in state["issues"]) or "なし",
            report=state["report"],
        )
        if state.get("feedback"):
            prompt += REVIEWER_FOLLOWUP.format(feedback=state["feedback"])
        result, cost = structured(Review, prompt, "reviewer")
        entry = {"round": state["revisions"], "approved": result.approved, "feedback": result.feedback}
        return {
            "approved": result.approved,
            "feedback": result.feedback,
            "history": state.get("history", []) + [entry],
            "costs": [cost],
        }

    def route_after_router(state: State) -> Literal["extractor", "__end__"]:
        return "extractor" if state["doc_type"] == "commercial_invoice" else END

    def route_after_review(state: State) -> Literal["writer", "__end__"]:
        if state["approved"] or state["revisions"] >= MAX_REVISIONS:
            return END
        return "writer"

    g = StateGraph(State)
    for name, fn in [("router", router), ("extractor", extractor), ("calc_check", calc_check),
                     ("regulation", regulation), ("writer", writer), ("reviewer", reviewer)]:
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router)
    g.add_edge("extractor", "calc_check")
    g.add_edge("extractor", "regulation")
    g.add_edge(["calc_check", "regulation"], "writer")  # 両方終わるのを待って合流
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review)
    return g.compile()
