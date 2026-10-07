"""教练 Agent 节点（单 Agent 架构，技术设计文档（新）§5/§7）。

情景演练六节点保留在 app/agents/roleplay/（§5 目录约定）；
本包承载新增的 briefing（访前简报）节点。
"""
from app.agents.coach.briefing import make_briefing_node

__all__ = ["make_briefing_node"]
