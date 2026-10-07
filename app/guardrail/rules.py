"""独立合规规则加载与校验；不执行检测、语义调用、改写或写库。"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

DEFAULT_RULE_PATH = Path(__file__).resolve().parents[2] / "data/compliance/rules.yaml"
RULE_IDS = {f"CMP-{i:02}" for i in range(1, 11)}


class RuleLoadError(ValueError):
    """规则文件加载失败，错误不回显文件正文。"""


class RulePatterns(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    keywords: tuple[str, ...]
    regex: tuple[str, ...]
    semantic: str = Field(min_length=1)
    context_keywords: tuple[str, ...] = ()
    reuse_rules: tuple[str, ...] = ()

    @field_validator("keywords", "context_keywords", "reuse_rules")
    @classmethod
    def nonempty_items(cls, value):
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("empty or duplicate pattern item")
        return value

    @field_validator("regex")
    @classmethod
    def valid_regex(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("duplicate regex")
        for expression in value:
            if not expression.strip():
                raise ValueError("empty regex")
            try:
                re.compile(expression)
            except re.error:
                raise ValueError("invalid regex") from None
        return value

    def compiled_regex(self) -> tuple[re.Pattern, ...]:
        """为后续引擎提供已验证的 Python 正则，不解释为命令。"""
        return tuple(re.compile(expression) for expression in self.regex)


class ComplianceRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str = Field(pattern=r"^CMP-(0[1-9]|10)$")
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["warn", "block", "rewrite"]
    action: Literal["log", "block", "rewrite", "escalate"]
    action_description: str = Field(min_length=1)
    applies_to: Literal["both", "agent_output"]
    patterns: RulePatterns
    violation_example: str = Field(min_length=1)
    compliant_response: str = Field(min_length=1)
    related_hooks: tuple[str, ...]


class ComplianceRuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    rules: tuple[ComplianceRule, ...]

    @model_validator(mode="after")
    def check_set(self):
        if len(self.rules) != 10 or {r.rule_id for r in self.rules} != RULE_IDS:
            raise ValueError("requires exactly CMP-01 through CMP-10")
        for rule in self.rules:
            expected = "block" if rule.rule_id in {"CMP-05", "CMP-06"} else "rewrite" if rule.rule_id == "CMP-10" else "warn"
            if rule.severity != expected:
                raise ValueError("severity mismatch")
            if rule.rule_id == "CMP-10":
                if rule.action != "rewrite" or rule.applies_to != "agent_output":
                    raise ValueError("output rule contract mismatch")
                if set(rule.patterns.reuse_rules) != RULE_IDS - {"CMP-10"} or len(rule.patterns.reuse_rules) != 9:
                    raise ValueError("output rule must reference CMP-01 through CMP-09")
                if rule.patterns.keywords or rule.patterns.regex:
                    raise ValueError("output rule uses referenced patterns")
            else:
                if rule.applies_to != "both" or rule.patterns.reuse_rules:
                    raise ValueError("input rule contract mismatch")
                if not rule.patterns.keywords and not rule.patterns.regex:
                    raise ValueError("input rule requires lexical patterns")
            if rule.rule_id == "CMP-05" and not rule.patterns.context_keywords:
                raise ValueError("document rule requires context gate")
        return self

    def by_id(self, rule_id: str) -> ComplianceRule:
        return next(rule for rule in self.rules if rule.rule_id == rule_id)


def load_compliance_rules(path: str | Path | None = None) -> ComplianceRuleSet:
    """从独立 YAML 加载，不读 config.yaml、不调用图或向量入库。"""
    rule_path = Path(path) if path is not None else DEFAULT_RULE_PATH
    try:
        payload = yaml.safe_load(rule_path.read_text(encoding="utf-8-sig"))
        return ComplianceRuleSet.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RuleLoadError(f"规则文件校验失败 ({rule_path.name}): {type(exc).__name__}") from None
