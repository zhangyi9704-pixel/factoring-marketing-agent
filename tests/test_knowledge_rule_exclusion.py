"""规则态 YAML 必须绕过知识目录入库，不触发知识加载或外部服务。"""
from app.config import Settings
from app.rag.ingest import KnowledgeIngestor, DocResult


def test_rules_yaml_excluded_from_category_ingest(tmp_path):
    directory = tmp_path / "data/compliance"
    directory.mkdir(parents=True)
    knowledge = directory / "KB-CMP-001.md"
    knowledge.write_text("知识态测试文件", encoding="utf-8")
    (directory / "rules.yaml").write_text("rules:\n  - rule_id: CMP-01\n", encoding="utf-8")
    (directory / "rules.yml").write_text("rules: []", encoding="utf-8")
    visited = []
    ingestor = KnowledgeIngestor(Settings(_env_file=None), project_root=tmp_path,
                                embedder=object(), store=object(), session_factory=lambda: None)

    def record(path, *, source, priority):
        visited.append(path.name)
        return DocResult(path.name, source, "indexed", chunk_cnt=2)

    ingestor.ingest_file = record
    results = ingestor.ingest_source("compliance")
    assert visited == [knowledge.name]
    assert len(results) == 1 and results[0].status == "indexed"
