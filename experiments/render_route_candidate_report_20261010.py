"""Render a teacher-friendly, side-by-side case viewer from scored outputs."""
from __future__ import annotations

import html

import task4_core as core
from experiments.run_deepseek_route_candidate_20261010 import ROOT

SCORES = ROOT / "outputs/deepseek_route_candidate_20261010/score/all_scores.json"
OUT = ROOT / "docs/赛题4_DeepSeek路由与候选链_30题逐题横排_2026-10-10.html"
FEATURED = {"energy_environment_04_Q02", "economy_trade_074_Q09",
            "diplomacy_009_Q07", "public_safety_129_Q09"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def box(title: str, answer: object, chains: object, score: object = None) -> str:
    display = "缺失或无效" if answer is None else str(answer)
    chain_text = "—" if chains is None else str(chains)
    metric = "" if score is None else (
        f"<small>答案字符 F1 {score['answer_char_f1']:.2f} · "
        f"完整链 {int(score['chain_exact'])}</small>")
    return (f"<section class='box'><h4>{esc(title)}</h4>"
            f"<p>{esc(display)}</p><code>{esc(chain_text)}</code>{metric}</section>")


def render_case(case: dict) -> str:
    v4, method = case["v4"], case["method"]
    route = case["route"] or {}
    change = (case["metrics"]["method"]["answer_char_f1"] -
              case["metrics"]["v4"]["answer_char_f1"])
    gold = box("原始 gold", case["gold_answer"], case["gold_chains"])
    baseline = box("DeepSeek V4 初答", v4["answer"] if v4 else None,
                   v4["evidence_chain"] if v4 else None, case["metrics"]["v4"])
    revised = box("路由＋候选链流程", method["answer"] if method else None,
                  method["evidence_chain"] if method else None,
                  case["metrics"]["method"])
    opened = " open" if case["sample_id"] in FEATURED else ""
    return (f"<details{opened}><summary><strong>{esc(case['sample_id'])}</strong> "
            f"<span>{esc(case['view'])} · {esc(case['reasoning_type'])} · "
            f"路由 {esc(route.get('route', '无效'))} · "
            f"答案 F1 变化 {change:+.2f} · {esc(case['decision'])}</span></summary>"
            f"<p class='question'>{esc(case['question'])}</p>"
            f"<div class='columns'>{gold}{baseline}{revised}</div></details>")


def main() -> None:
    report = core.read_json(SCORES)
    all_summary = report["summary"]["all"]
    v4, method = all_summary["v4"], all_summary["method"]
    rows = "\n".join(render_case(case) for case in report["cases"])
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>赛题4：DeepSeek 路由与候选链 30 题配对复核</title>
<style>
body{{font:16px/1.6 system-ui,'Microsoft YaHei',sans-serif;color:#172337;background:#f3f6fa;margin:0 auto;padding:32px;max-width:1500px}}
h1{{font-size:1.8rem;margin:0 0 8px}}h2{{font-size:1.3rem;margin-top:34px}}
.intro,.note{{color:#48576e;max-width:1000px}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin:24px 0}}
.metric{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:16px}}
.metric b{{display:block;font-size:1.5rem}}.metric em{{font-style:normal;color:#b42318}}
details{{background:white;border:1px solid #dce4ed;border-radius:12px;margin:12px 0;padding:12px 16px}}
summary{{cursor:pointer;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}}summary span{{color:#566579;font-size:.9rem}}
.question{{font-weight:650;background:#edf3fc;border-radius:8px;padding:10px 14px}}
.columns{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.box{{border:1px solid #e0e7ef;border-radius:9px;padding:12px;min-width:0}}
.box h4{{margin:0 0 8px;color:#1b4b81}}.box p{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 12px}}
code{{display:block;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;border-radius:5px;padding:8px}}
small{{display:block;color:#566579;margin-top:8px}}@media(max-width:900px){{.columns,.metrics{{grid-template-columns:1fr}}}}
</style><body>
<h1>先判路线、再给候选链：DeepSeek 30 题配对复核</h1>
<p class="intro">同一冻结 fit 题集；V4 为现成初答，新流程在其上增加路线判断、合法 ID 检查、候选链和条件复核。点击题目展开。下列分数是训练 gold 本地代理指标，不是官方测试分。</p>
<div class="metrics">
<div class="metric">有效输出<b>{v4['valid']} → <em>{method['valid']}</em> /30</b></div>
<div class="metric">完整链匹配<b>{round(v4['chain_exact']*30)} → <em>{round(method['chain_exact']*30)}</em> /30</b></div>
<div class="metric">答案字符 F1<b>{v4['answer_char_f1']:.3f} → <em>{method['answer_char_f1']:.3f}</em></b></div>
<div class="metric">严格拒答命中<b>9 → <em>6</em> /12</b></div>
</div>
<p class="note">负结果保留：新增 49 次 DeepSeek 请求，输入 {report['usage']['prompt_tokens']:,} token、输出 {report['usage']['completion_tokens']:,} token。候选池只提供备选，模型仍须回查完整原文；本实验实际没有提高链选择。</p>
<h2>逐题横排：原始 gold／V4 初答／新流程</h2>{rows}
</body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
