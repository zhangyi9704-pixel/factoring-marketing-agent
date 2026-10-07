"""用户成绩画像（TDD §20 Phase 4 主管视角 `GET /users/{id}/scores`）。

跨会话聚合评估数据（只读 DB，不调 LLM）：
- 该用户全部会话与已评估会话数；
- 机会诊断均分 / 能力评分均分；
- 机会等级分布（高潜机会 / 观察 / 暂缓）；
- 各评分维度（机会 5 维 + 能力 4 维）的跨会话均值；
- 最近评估时间线；
- **辅导板块 coaching**（五段式，维度标签复用 evaluation.common.DIM_LABELS）：
  一、机会诊断（均分/等级分布/逐维均值）
  二、复盘辅导（能力均分/逐维均值）
  三、评分证据（跨会话逐维证据条数 + 最近一次评估的原话证据）
  四、遗漏问题（行业核查清单项按出现次数聚合）
  五、改善空间以及下一步行动（均分最低维度 + 证据缺口维度 + 最近行动清单）

只聚合**已落库**的评估（t_evaluation / t_evaluation_evidence），不懒计算，
避免主管看板触发一串重 LLM 调用。空数据返回结构完整、数值置 0/空列表的
画像（不报错）。
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings, get_settings
from app.evaluation.common import DIM_LABELS

logger = logging.getLogger(__name__)

LEVELS = ("高潜机会", "观察", "暂缓")

WEAKEST_TOP_N = 3          # 改善空间：均分最低 / 证据缺口最多的维度条数


class UserScoreServiceError(RuntimeError):
    """用户成绩画像错误。"""


class UserScoreService:
    """主管画像聚合（注入式，只读 DB）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        session_service: object = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._session_service = session_service

    @property
    def session_service(self):
        if self._session_service is None:
            from app.services.session_service import SessionService
            self._session_service = SessionService(self.settings)
        return self._session_service

    def scores(self, user_id: int) -> dict:
        """返回该用户的成绩画像（聚合画像 + 五段式辅导板块）。"""
        report: dict | None = None
        coaching: dict = {}
        for event in self.scores_stream(user_id):
            if event["type"] == "profile":
                report = dict(event["data"])
            elif event["type"] == "section":
                coaching[event["key"]] = event["data"]
        report["coaching"] = coaching
        return report

    def scores_stream(self, user_id: int):
        """流式版本（生成器）：profile 帧 → 五个 section 帧 → done 帧。

        帧协议见 app/api/sse.py；三段（评分证据）在产出时才读
        t_evaluation_evidence，画像主体先行到达。
        """
        sessions = self.session_service.list_sessions_by_user(user_id)
        evaluated = self._collect_evaluated(user_id, sessions)
        evidence_ids = [
            e["evaluation_id"] for e in evaluated if e["evaluation_id"] is not None
        ]

        opp_scores = [e["opportunity_score"] for e in evaluated if e["opportunity_score"] is not None]
        sales_scores = [e["sales_score"] for e in evaluated if e["sales_score"] is not None]
        level_dist: dict[str, int] = {level: 0 for level in LEVELS}
        dim_sums: dict[str, float] = {}
        dim_counts: dict[str, int] = {}
        for e in evaluated:
            if e["level"] in level_dist:
                level_dist[e["level"]] += 1
            for key, val in e["dims"].items():
                dim_sums[key] = dim_sums.get(key, 0.0) + val
                dim_counts[key] = dim_counts.get(key, 0) + 1
        dimension_averages = {
            key: round(dim_sums[key] / dim_counts[key], 2) for key in dim_sums
        }
        latest = sorted(
            evaluated, key=lambda e: e["generated_at"] or "", reverse=True,
        )
        yield {"type": "profile", "data": {
            "user_id": user_id,
            "total_sessions": len(sessions),
            "evaluated_sessions": len(evaluated),
            "avg_opportunity": _mean(opp_scores),
            "avg_sales_score": _mean(sales_scores),
            "level_distribution": level_dist,
            "dimension_averages": dimension_averages,
            "latest_sessions": [_public_summary(e) for e in latest[:10]],
        }}
        for key, section in self._coaching_sections(
            evaluated, level_dist, dimension_averages, evidence_ids,
        ):
            yield {"type": "section", "key": key, "data": section}
        yield {"type": "done"}

    # -------------------------------------------------
    # 内部
    # -------------------------------------------------

    def _collect_evaluated(self, user_id: int, sessions: list[dict]) -> list[dict]:
        """对每个已结束且已评估的会话抽取摘要。"""
        from app.models import Evaluation

        factory = self.session_service.session_factory
        evaluated: list[dict] = []
        with factory() as session:
            for s in sessions:
                if s.get("status") != "finished":
                    continue
                ev = session.query(Evaluation).filter(
                    Evaluation.session_id == s["session_id"],
                ).one_or_none()
                if ev is None:
                    continue
                evaluated.append(self._summarize(ev, s))
        return evaluated

    def _load_evidence(self, evaluation_ids: list[int]) -> dict[int, list[dict]]:
        """按 evaluation_id 读回证据原话行（辅导板块三）。"""
        if not evaluation_ids:
            return {}
        from app.models import EvaluationEvidence

        grouped: dict[int, list[dict]] = {}
        try:
            with self.session_service.session_factory() as session:
                rows = session.query(EvaluationEvidence).filter(
                    EvaluationEvidence.evaluation_id.in_(evaluation_ids),
                ).order_by(EvaluationEvidence.evaluation_id, EvaluationEvidence.id).all()
        except Exception as exc:                        # 证据缺失不影响画像主体
            logger.warning("主管看板证据读取失败，辅导板块证据段置空: %s", exc)
            return {}
        for row in rows:
            grouped.setdefault(int(row.evaluation_id), []).append({
                "dimension": row.dimension,
                "turn": row.turn,
                "quote_sales": row.quote_sales,
                "quote_customer": row.quote_customer,
                "comment": row.comment,
                "score": float(row.score) if row.score is not None else None,
            })
        return grouped

    @staticmethod
    def _summarize(ev, session_row: dict) -> dict:
        opp = ev.opportunity or {}
        sales = ev.sales_score or {}
        opp_res = opp.get("dimension_scores") or {}
        sales_res = sales.get("dimension_scores") or {}
        dims: dict[str, float] = {}
        opp_dims: dict[str, float] = {}
        sales_dims: dict[str, float] = {}
        for key, res in opp_res.items():
            _put_dim(dims, key, res)
            _put_dim(opp_dims, key, res)
        for key, res in sales_res.items():
            _put_dim(dims, key, res)
            _put_dim(sales_dims, key, res)
        missing_flags = {
            key for key, res in {**opp_res, **sales_res}.items()
            if isinstance(res, dict) and res.get("evidence_missing")
        }
        return {
            "session_id": session_row["session_id"],
            "case_id": session_row.get("case_id", ""),
            "opportunity_score": opp.get("score"),
            "level": opp.get("level"),
            "sales_score": float(sales.get("total")) if sales.get("total") is not None else None,
            "generated_at": _iso(ev.created_at),
            "dims": dims,
            # —— 辅导板块聚合原料（不出 API latest_sessions，见 _public_summary）——
            "evaluation_id": getattr(ev, "id", None),
            "opp_dims": opp_dims,
            "sales_dims": sales_dims,
            "evidence_missing": missing_flags,
            "missing_items": (ev.missing or {}).get("items") or [],
            "next_actions": (ev.next_action or {}).get("actions") or [],
        }

    # -------------------------------------------------
    # 辅导板块（五段式，§20 主管视角；逐段产出供流式端点）
    # -------------------------------------------------

    def _coaching_sections(
        self,
        evaluated: list[dict],
        level_dist: dict[str, int],
        dimension_averages: dict[str, float],
        evidence_ids: list[int],
    ):
        """逐段 yield (key, section)：一机会诊断 / 二复盘辅导 / 三评分证据 /
        四遗漏问题 / 五改善空间以及下一步行动（全部规则聚合，零 LLM）。

        三段产出时才读证据表（_load_evidence），使画像主体先行到达。
        """
        by_time = sorted(evaluated, key=lambda e: e["generated_at"] or "", reverse=True)
        latest_evaluated = by_time[0] if by_time else None

        yield "opportunity", {                           # 一、机会诊断
            "avg_score": _mean([e["opportunity_score"] for e in evaluated
                                if e["opportunity_score"] is not None]),
            "level_distribution": level_dist,
            "dimension_averages": _dim_list(_avg_dims(evaluated, "opp_dims")),
        }

        yield "review", {                                # 二、复盘辅导
            "avg_total": _mean([e["sales_score"] for e in evaluated
                                if e["sales_score"] is not None]),
            "dimension_averages": _dim_list(_avg_dims(evaluated, "sales_dims")),
        }

        evidence_map = self._load_evidence(evidence_ids)

        # 四、遗漏问题：跨会话按清单项计数（出现次数 = 多次没问到的稳定短板）
        missing_counts: dict[str, dict] = {}
        for e in evaluated:
            for m in e["missing_items"]:
                item = str(m.get("item") or "").strip()
                if not item:
                    continue
                entry = missing_counts.setdefault(item, {
                    "item": item, "count": 0,
                    "impact": str(m.get("impact") or ""), "suggestion": str(m.get("suggestion") or ""),
                })
                entry["count"] += 1

        # 五、证据缺口：维度证据缺失次数（缺证据 = 该维提问没有触达）
        gap_counts: dict[str, int] = {}
        for e in evaluated:
            for key in e["evidence_missing"]:
                gap_counts[key] = gap_counts.get(key, 0) + 1

        # 三、证据统计：跨会话逐维证据条数 + 最近一次评估的原话明细
        evidence_quotes: dict[str, int] = {}
        for rows in evidence_map.values():
            for row in rows:
                dim = str(row.get("dimension") or "")
                if dim:
                    evidence_quotes[dim] = evidence_quotes.get(dim, 0) + 1
        latest_evidence = {"session_id": "", "case_id": "", "generated_at": "", "items": []}
        for e in by_time:                                # 最近一次有证据明细的评估
            rows = evidence_map.get(e["evaluation_id"]) if e["evaluation_id"] is not None else None
            if rows:
                latest_evidence = {
                    "session_id": e["session_id"], "case_id": e["case_id"],
                    "generated_at": e["generated_at"],
                    # 评分证据按轮次合并：一轮只说一次对话（销售/客户引语），
                    # 下面列出命中的多个维度（含分值），并按 turn 升序还原问答顺序，
                    # 避免同轮多条原话重复显示。
                    "items": _group_evidence_by_turn(rows),
                }
                break

        yield "evidence", {                              # 三、评分证据
            "by_dimension": [
                {"key": k, "label": _label(k), "quotes": v}
                for k, v in sorted(evidence_quotes.items(), key=lambda kv: -kv[1])
            ],
            "latest": latest_evidence,
        }

        yield "missing", sorted(                         # 四、遗漏问题
            missing_counts.values(), key=lambda m: -m["count"],
        )

        yield "improvement", {                           # 五、改善空间以及下一步行动
            "weakest_dimensions": [
                {"key": k, "label": _label(k), "avg": v}
                for k, v in sorted(dimension_averages.items(), key=lambda kv: kv[1])
                [:WEAKEST_TOP_N]
            ],
            "evidence_gaps": [
                {"key": k, "label": _label(k), "sessions": v}
                for k, v in sorted(gap_counts.items(), key=lambda kv: -kv[1])
                [:WEAKEST_TOP_N]
            ],
            "next_actions": list(latest_evaluated["next_actions"]) if latest_evaluated else [],
        }


# -----------------------------------------------------
# 组装小工具
# -----------------------------------------------------

def _public_summary(e: dict) -> dict:
    """latest_sessions 对外字段（聚合原料不出 API）。"""
    return {
        "session_id": e["session_id"],
        "case_id": e["case_id"],
        "opportunity_score": e["opportunity_score"],
        "level": e["level"],
        "sales_score": e["sales_score"],
        "generated_at": e["generated_at"],
    }


def _label(key: str) -> str:
    return DIM_LABELS.get(key, key)


def _dim_list(avg_dict: dict[str, float]) -> list[dict]:
    return [{"key": k, "label": _label(k), "avg": v} for k, v in avg_dict.items()]


def _avg_dims(evaluated: list[dict], field: str) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for e in evaluated:
        for key, val in e[field].items():
            sums[key] = sums.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
    return {key: round(sums[key] / counts[key], 2) for key in sums}


def _group_evidence_by_turn(rows: list[dict]) -> list[dict]:
    """把证据行按轮次合并：一轮一次对话，下面列出命中的多个维度。

    同一轮的多条维度证据分享同一段销售/客户引语；合并后一项展示一轮对话，
    避免"同轮原话重复显示"，并按 turn 升序还原问答顺序。
    """
    groups: dict[int, dict] = {}
    order: list[int] = []
    for row in rows:
        turn = row.get("turn") if isinstance(row.get("turn"), int) else int(row.get("turn") or 0)
        if turn not in groups:
            groups[turn] = {"turn": turn, "quote_sales": "", "quote_customer": "", "dimensions": []}
            order.append(turn)
        g = groups[turn]
        if not g["quote_sales"] and (row.get("quote_sales") or "").strip():
            g["quote_sales"] = row["quote_sales"]
        if not g["quote_customer"] and (row.get("quote_customer") or "").strip():
            g["quote_customer"] = row["quote_customer"]
        g["dimensions"].append({
            "label": _label(str(row.get("dimension") or "")),
            "score": row.get("score"),
        })
    return [groups[t] for t in sorted(order)]


def _put_dim(container: dict[str, float], key: str, res: dict) -> None:
    """dimension_scores 条目 → 扁平维名 → 值（gated_score 优先，落 score）。"""
    value = res.get("gated_score") if res.get("gated_score") is not None else res.get("score")
    if value is not None:
        container[str(key)] = float(value)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _iso(value) -> str:
    return value.isoformat(timespec="seconds") if value else ""
