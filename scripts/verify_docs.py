"""Read-only local documentation links, command paths and PowerShell syntax check."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["README.md", "docs/运行手册.md", "docs/系统使用讲解与验收演示话术.md", "docs/建筑行业验收演示话术.md", "docs/文档收尾与评测修复验收报告.md", "docs/开发手册.md", "docs/验收报告_2026-10-07.md", "docs/交接清单_2026-10-07.md"]
TARGETS += ["docs/reference/" + n for n in ("Agent_requirement_document（PRD）.md", "技术设计文档（TDD）.md", "技术设计文档（新）.md", "开发总结.md")]
TARGETS += ["archive/README.md", "docs/项目完整性审查与归档报告_2026-10-07.md",
            "archive/docs/阈值校准记录-修复前快照.md", "archive/docs/阈值校准记录-中点方案未通过.md",
            "docs/指标评测验收报告.md", "docs/前端鉴权与浏览器验收报告.md",
            "docs/审核台前端与浏览器验收报告.md", "docs/知识回流与发布验收报告.md",
            "docs/会话恢复并发与SSE验收报告.md"]

def check_link(source, target):
    target = target.strip().strip("<>")
    if urlsplit(target).scheme or target.startswith("#"):
        return True
    relative = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return (source.parent / relative).exists()

def main():
    errors, blocks, links, commands = [], [], 0, 0
    for name in TARGETS:
        p = ROOT / name
        if not p.is_file():
            errors.append(f"missing document: {name}")
            continue
        s = p.read_text(encoding="utf-8-sig")
        # Historical command text is preserved; only the appended errata is checked.
        if name == "docs/开发手册.md":
            s = s.split("## 30. M6修复与文档收尾", 1)[-1]
        if "/reference/" in name:
            s = s.split("## 2026-10-07 当前项目勘误", 1)[-1]
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", s):
            links += 1
            if not check_link(p, target):
                errors.append(f"broken local link in {name}: {target}")
        for path in re.findall(r"(?:archive[/\\]tools|scripts|deploy)[/\\][\w/\\.-]+\.(?:py|ps1|yaml)", s):
            commands += 1
            if not (ROOT / path.replace("\\", "/")).is_file():
                errors.append(f"missing command path in {name}: {path}")
        blocks.extend(re.findall(r"```powershell\s*\n(.*?)```", s, re.S))
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        errors.append("PowerShell parser unavailable")
    else:
        # Pass strings over stdin and parse; never execute documentation commands.
        parser = "$texts=($input | Out-String | ConvertFrom-Json); $bad=0; foreach($text in $texts) { $tokens=$null; $errs=$null; [void][System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errs); $bad += @($errs).Count }; Write-Output $bad"
        proc = subprocess.run([shell, "-NoProfile", "-Command", parser], input=json.dumps(blocks), text=True, capture_output=True)
        if proc.returncode or proc.stdout.strip() != "0":
            errors.append("PowerShell code block syntax validation failed")
        for script in (ROOT / "scripts").glob("start_local.ps1"):
            proc = subprocess.run([shell, "-NoProfile", "-Command", parser], input=json.dumps([script.read_text(encoding="utf8")]), text=True, capture_output=True)
            if proc.returncode or proc.stdout.strip() != "0":
                errors.append("start_local.ps1 syntax validation failed")
    proof = json.loads((ROOT / "docs/validation/reference-errata-prefix.json").read_text(encoding="utf8"))
    for row in proof:
        prefix = (ROOT / row["file"]).read_bytes()[:row["original_bytes"]]
        if hashlib.sha256(prefix).hexdigest() != row["original_sha256"]:
            errors.append(f"historical prefix changed: {row['file']}")
    result = dict(documents=len(TARGETS), links=links, command_paths=commands, powershell_blocks=len(blocks), historical_prefixes=len(proof), errors=errors)
    print(json.dumps(result, ensure_ascii=False))
    return bool(errors)

if __name__ == "__main__":
    sys.exit(main())
