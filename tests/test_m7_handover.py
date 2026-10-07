"""Restore integrity and target guards without containers/cloud."""
import json
import pytest
from sqlalchemy import create_engine,text
from scripts.verify_handover import sql_fingerprints
from scripts import run_m7_validation


def test_restore_fingerprints_ignore_row_order_but_detect_content_loss():
    engine=create_engine("sqlite://")
    with engine.begin() as c:
        c.execute(text("CREATE TABLE sample (id INTEGER, value TEXT)"))
        c.execute(text("INSERT INTO sample VALUES (2,'版本二'),(1,'来源一')"))
    original=sql_fingerprints(engine)
    with engine.begin() as c:
        c.execute(text("DELETE FROM sample"))
        c.execute(text("INSERT INTO sample VALUES (1,'来源一'),(2,'版本二')"))
    assert sql_fingerprints(engine)==original
    with engine.begin() as c:c.execute(text("UPDATE sample SET value='changed' WHERE id=2"))
    assert sql_fingerprints(engine)!=original


@pytest.mark.parametrize("status,database,collection",[
    ("running","factoring_m7_abcdef123456","factoring_m7_abcdef123456"),
    ("passed_logical_restore","factoring_coach","factoring_m7_abcdef123456"),
    ("passed_logical_restore","factoring_m7_abcdef123456","factoring_knowledge_v2"),
])
def test_unverified_or_primary_targets_rejected(tmp_path,monkeypatch,status,database,collection):
    root=tmp_path/"project";folder=root/"docs/validation";folder.mkdir(parents=True)
    (folder/"m7-handover-abcdef123456.json").write_text(json.dumps(dict(status=status,database=database,collection=collection)),encoding="utf8")
    monkeypatch.setattr(run_m7_validation,"ROOT",root)
    with pytest.raises(ValueError):run_m7_validation.configure("abcdef123456")


def test_run_identifier_cannot_be_a_path_or_primary_name():
    for value in ("../outside","factoring_coach","bad;command"):
        with pytest.raises(ValueError):run_m7_validation.configure(value)
