"""Read-only source inventory and conservative v3 baseline migration (stdlib only)."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\BaiduNetdiskDownload\HM\保理营销教练Agent")
AUDIT = ROOT / "docs/reference/migration"
DIRS = ("app", "frontend", "cases", "data", "scripts", "tests")
FILES = ("config.yaml", "requirements.txt", "pytest.ini", "conftest.py")
ARCHIVE = ("docker-compose.yml", "技术设计文档（新）.md", "技术设计文档（TDD）.md",
           "开发总结.md", "Agent_requirement_document（PRD）.md")
MERGE = (".env.example", ".gitignore")
EXCLUDED = {".git", "__pycache__", ".pytest_cache", "logs", ".env", ".idea",
            ".claude", ".workbuddy", ".venv", "node_modules", "volumes", "docker_data"}
REQUIRED = ("init_tables.py", "init_mysql.sql", "create_milvus_collection.py",
            "ingest_knowledge.py", "init_cases.py", "verify_milvus.py",
            "verify_deepseek.py", "verify_briefing.py", "verify_roleplay.py",
            "verify_evaluation.py", "verify_retrieval.py", "verify_phase4.py", "embed_server.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded(path):
    return (bool(set(path.parts) & EXCLUDED) or path.suffix.lower() in {".pyc", ".pyo", ".log"}
            or (path.name.startswith(".env.") and path.name != ".env.example"))


def migrated_bytes(name):
    content = (SOURCE / name).read_bytes()
    if name == "技术设计文档（TDD）.md":
        for key, value in source_credentials():
            if key == "DEEPSEEK_API_KEY":
                content = content.replace(value, b"${DEEPSEEK_API_KEY}")
    return content


def source_credentials():
    values = []
    for line in (SOURCE / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if re.search(r"KEY|PASSWORD|SECRET|TOKEN", key) and len(value) >= 8 and "${" not in value:
            values.append((key.strip(), value.encode()))
    return values


def inventory():
    rows, omissions = [], []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        name = rel.as_posix()
        if excluded(rel):
            omissions.append({"path": name, "reason": "excluded"})
            continue
        if rel.parts[0] in DIRS or name in FILES:
            dest, action = name, "copy"
        elif name in ARCHIVE:
            dest, action = "docs/reference/" + name, "archive"
        elif name in MERGE:
            dest, action = name, "merge"
        else:
            omissions.append({"path": name, "reason": "outside requested scope"})
            continue
        assert not path.is_symlink(), f"Symlink requires review: {name}"
        target = ROOT / dest
        rows.append({"source": name, "target": dest, "action": action,
                     "source_sha256": digest(path), "bytes": path.stat().st_size,
                     "expected_target_sha256": hashlib.sha256(migrated_bytes(name)).hexdigest(),
                     "target_sha256_before": digest(target) if target.exists() else None})
    for name in DIRS + FILES + ARCHIVE + MERGE:
        assert (SOURCE / name).exists(), f"Missing required source: {name}"
    return rows, omissions


def secret_scan(rows):
    # Never print values. Compare copied data against source credential literals.
    literals = [value for key, value in source_credentials()]
    hits = []
    for row in rows:
        if row["action"] == "merge":
            continue
        content = migrated_bytes(row["source"])
        if (any(value in content for value in literals)
                or re.search(rb"\bsk-[A-Za-z0-9_-]{24,}\b", content)
                or re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content)):
            hits.append(row["source"])
    assert not hits, "Credential candidates (values hidden): " + ", ".join(hits)


def dump(name, value):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plan():
    assert not (AUDIT / "plan.json").exists(), "Existing migration plan retained; use verify"
    rows, omissions = inventory()
    secret_scan(rows)
    conflicts = [r for r in rows if r["target_sha256_before"] is not None
                 and r["source_sha256"] != r["target_sha256_before"]]
    preserved = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {".git", ".venv", "app", "frontend", "cases", "data", "scripts", "tests"}:
            continue
        if rel.as_posix().startswith("docs/reference/migration/"):
            continue
        preserved[rel.as_posix()] = digest(path)
    status = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(SOURCE),
                             "status", "--short"], check=True, capture_output=True).stdout.decode("utf-8")
    dump("plan.json", {"source_root": str(SOURCE), "target_root": str(ROOT), "rows": rows,
                       "omissions": omissions, "conflicts": conflicts, "preserved": preserved,
                       "source_git_status": status})
    print(f"PLAN: copy/archive={sum(r['action'] != 'merge' for r in rows)}, merge=2, conflicts={len(conflicts)}")
    for row in conflicts:
        print("CONFLICT: " + row["target"] + " (hashes recorded; values hidden)")
    print("PASS: source credential literal / token / private-key scan")


def apply():
    saved = json.loads((AUDIT / "plan.json").read_text(encoding="utf-8"))
    assert all(r["action"] == "merge" for r in saved["conflicts"]), "Copy conflicts require review"
    for row in saved["rows"]:
        src, dest = SOURCE / row["source"], ROOT / row["target"]
        assert digest(src) == row["source_sha256"], f"Source changed: {row['source']}"
        actual = digest(dest) if dest.exists() else None
        allowed = {row["target_sha256_before"]}
        if row["action"] != "merge":
            allowed.add(row["expected_target_sha256"])
        assert actual in allowed, f"Target changed: {row['target']}"
    secret_scan(saved["rows"])
    count = 0
    for row in saved["rows"]:
        if row["action"] == "merge":
            continue
        dest = ROOT / row["target"]
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if row["source_sha256"] != row["expected_target_sha256"]:
                dest.write_bytes(migrated_bytes(row["source"]))
            else:
                shutil.copy2(SOURCE / row["source"], dest)
            count += 1
    # Preserve every M0 template value. Legacy host aliases are unused by Settings;
    # they are intentionally not resurrected (see docs/环境变量对照表.md).
    source_keys = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", (SOURCE / ".env.example").read_text(encoding="utf-8"), re.M))
    target_keys = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
    aliases = source_keys - target_keys
    assert aliases <= {"MILVUS_HOST", "MILVUS_PORT", "MILVUS_METRICS_PORT", "REDIS_HOST", "MYSQL_HOST"}
    ignore = ROOT / ".gitignore"
    existing = ignore.read_text(encoding="utf-8").splitlines()
    rules = [line for line in (SOURCE / ".gitignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#") and line not in existing]
    rules += [rule for rule in (".idea/", ".claude/", ".workbuddy/") if rule not in existing + rules]
    if rules:
        with ignore.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n# v3 migration: temporary knowledge files and IDE metadata.\n" + "\n".join(rules) + "\n")
    total = sum(r["action"] != "merge" and r["target_sha256_before"] is None for r in saved["rows"])
    dump("apply.json", {"copied_files": total, "copied_this_invocation": count, "added_ignore_rules": rules,
                        "legacy_keys_not_added": sorted(aliases), "env_example": "M0 keys and values retained"})
    print(f"APPLY: copied_files_total={total}, copied_this_invocation={count}; added_ignore_rules={len(rules)}; M0 template retained")


def verify():
    saved = json.loads((AUDIT / "plan.json").read_text(encoding="utf-8"))
    current, omissions = inventory()
    assert [(r['source'], r['source_sha256']) for r in current] == [(r['source'], r['source_sha256']) for r in saved['rows']], "Source inventory changed"
    assert omissions == saved['omissions'], "Source exclusions changed"
    status = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(SOURCE),
                             "status", "--short"], check=True, capture_output=True).stdout.decode("utf-8")
    assert status == saved["source_git_status"], "Source git status changed"
    manifest = []
    for row in saved["rows"]:
        target_hash = digest(ROOT / row["target"])
        if row["action"] != "merge":
            assert target_hash == row["expected_target_sha256"], f"Hash mismatch: {row['target']}"
        manifest.append({"source": row["source"], "target": row["target"],
                         "source_sha256": row["source_sha256"], "target_sha256": target_hash,
                         "result": ("merged" if row["action"] == "merge" else
                                    "redacted" if row["source_sha256"] != target_hash else "identical")})
    # These two manuals are explicitly updated to record this step.
    editable = {".gitignore", "deploy/migrate_v3.py", "docs/开发手册.md", "docs/保理营销教练Agent-分步施工提示词.md"}
    preserved = {name: digest(ROOT / name) == sha for name, sha in saved["preserved"].items() if name not in editable}
    assert all(preserved.values()), "Existing protected files changed"
    banned = [p.relative_to(ROOT).as_posix() for name in DIRS for p in (ROOT / name).rglob("*")
              if excluded(p.relative_to(ROOT))]
    assert not banned, "Excluded artifacts migrated: " + ", ".join(banned)
    for name in REQUIRED:
        assert (ROOT / "scripts" / name).is_file(), f"Missing script: {name}"
    assert not (ROOT / "docker-compose.yml").exists()
    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".venv/", "__pycache__/", "*.py[cod]", ".pytest_cache/", "logs/",
            "*.log", "data/**/*.tmp", ".idea/", ".claude/", ".workbuddy/"} <= ignored
    secret_scan(saved["rows"])
    dump("manifest.json", manifest)
    dump("verification.json", {"identical_files": sum(r['result'] == 'identical' for r in manifest),
                               "redacted_files": sum(r['result'] == 'redacted' for r in manifest),
                               "merged_files": 2, "required_scripts": len(REQUIRED),
                               "protected_files_unchanged": preserved, "excluded_artifacts": banned,
                               "secret_scan": "no source credential literals, long sk tokens or private keys found"})
    print(f"PASS: {len(manifest) - 3} identical files; 1 redacted document; 2 merge decisions; {len(REQUIRED)} scripts present")
    print(f"PASS: {len(preserved)} protected files unchanged; no excluded artifacts; source unchanged")
    print("PASS: credential scan; root legacy compose absent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "apply", "verify"))
    mode = parser.parse_args().mode
    {"plan": plan, "apply": apply, "verify": verify}[mode]()
