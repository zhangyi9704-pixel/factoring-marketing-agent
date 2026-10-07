"""按 ORM 元数据创建全部业务表（幂等）。

用法：
    python scripts/init_tables.py            # 创建缺失的表（不删数据）
    python scripts/init_tables.py --drop     # 危险：先删除全部业务表再重建

目标库来自 .env / config.yaml（当前为本地 factoring_coach 库）。
本脚本保证新表结构与当前 ORM 模型一致；旧库仍需 README 所列增量迁移。
scripts/init_mysql.sql 是历史结构参考，不可用于当前初始化。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import inspect  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.db.mysql import get_mysql_database  # noqa: E402
from app.models import Base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="创建/重建业务表")
    parser.add_argument("--drop", action="store_true",
                        help="先删除全部业务表再重建（数据会丢失）")
    args = parser.parse_args()

    db = get_mysql_database()
    url = make_url(str(db.engine.url))
    print("=" * 60)
    print("业务表初始化（ORM 元数据）")
    print("=" * 60)
    print(f"目标库: {url.database}（口令与连接串不输出）")

    if args.drop:
        confirm = input("确认删除全部业务表？输入 yes 继续: ").strip().lower()
        if confirm != "yes":
            print("已取消")
            return 1
        Base.metadata.drop_all(db.engine)
        print("[OK] 已删除全部业务表")

    Base.metadata.create_all(db.engine)

    tables = sorted(t for t in inspect(db.engine).get_table_names() if t.startswith("t_"))
    expected = sorted(Base.metadata.tables)
    print(f"[OK] 业务表（{len(tables)} 个）: {', '.join(tables)}")
    if tables != expected:
        print(f"[WARN] 与 ORM 期望不一致，期望: {', '.join(expected)}")
        return 1
    print("结论: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
