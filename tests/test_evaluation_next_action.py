"""下一步行动建议单元测试（按机会等级分支，TDD §13.5）。"""
from app.evaluation.next_action import NextActionAdvisor


class TestNextActionAdvisor:

    def test_high_level_branch(self):
        actions = NextActionAdvisor().advise({"level": "高潜机会"}, missing_count=0)
        assert actions and actions[0]["priority"] == "high"
        assert any("尽调" in a["action"] for a in actions)

    def test_observe_branch(self):
        actions = NextActionAdvisor().advise({"level": "观察"}, missing_count=2)
        assert any("补齐" in a["action"] for a in actions)
        assert any("2" in a["note"] for a in actions)      # 引用遗漏数

    def test_pending_branch(self):
        actions = NextActionAdvisor().advise({"level": "暂缓"}, missing_count=5)
        assert any("信息不足" in a["action"] for a in actions)
        assert all(a["priority"] in ("high", "medium") for a in actions)

    def test_unknown_level_defaults_pending(self):
        actions = NextActionAdvisor().advise({"level": "未知"})
        assert actions and any("信息不足" in a["action"] for a in actions)
