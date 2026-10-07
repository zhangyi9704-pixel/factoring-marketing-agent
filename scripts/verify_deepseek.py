"""DeepSeek 连通性验证脚本（TDD §7 验证）。

用法：
    python scripts/verify_deepseek.py

验证内容：
    1. API Key 占位符解析与客户端初始化
    2. GET /models 可用模型列表，校验 config.yaml 配置的模型名
    3. chat / reasoning 双模型各发起一次极小真实请求
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.services.llm_service import LLMService, LLMServiceError  # noqa: E402


def mask(key: str | None) -> str:
    if not key:
        return "<未配置>"
    return f"{key[:6]}...{key[-4:]}"


def main() -> int:
    settings = get_settings()
    llm = settings.llm
    print("=" * 60)
    print("DeepSeek 连通性验证")
    print("=" * 60)
    print(f"base_url              : {llm.api.base_url}")
    print(f"api_key               : {mask(llm.api.api_key)}")
    print(f"reasoning 模型        : {llm.models.reasoning.name}")
    print(f"chat 模型             : {llm.models.chat.name}")
    print(f"timeout connect/read  : {llm.timeout.connect}/{llm.timeout.read}s")
    print(f"retry max_attempts    : {llm.retry.max_attempts}")
    print("-" * 60)

    try:
        svc = LLMService(settings)
        print("[OK] 客户端初始化成功（API Key 已解析）")
    except LLMServiceError as exc:
        print(f"[FAIL] 客户端初始化失败: {exc}")
        return 1

    # 1. 模型列表
    try:
        models = svc.list_models()
        print(f"[OK] 可用模型（{len(models)} 个）: {', '.join(models)}")
    except Exception as exc:
        print(f"[FAIL] 获取模型列表失败: {exc}")
        return 1

    for role, cfg_name in (
        ("reasoning", llm.models.reasoning.name),
        ("chat", llm.models.chat.name),
    ):
        if cfg_name in models:
            print(f"[OK] {role:9s} 配置模型 {cfg_name} 在可用列表中")
        else:
            print(f"[WARN] {role:9s} 配置模型 {cfg_name} 不在可用列表中（若调用报错请核对模型名）")

    # 2. 双模型实调（极小请求）
    print("-" * 60)
    all_ok = True
    for role in ("chat", "reasoning"):
        # reasoning 角色可能消耗较多推理 token，放宽 max_tokens
        limit = 512 if role == "reasoning" else 64
        try:
            t0 = time.time()
            reply = svc.chat("请只回复四个字：连接正常", model_role=role, max_tokens=limit)
            cost = time.time() - t0
            text = (reply or "").strip().replace("\n", " ")
            print(f"[OK] {role:9s} 调用成功（{cost:.1f}s）: {text[:60]}")
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] {role:9s} 调用失败: {exc}")

    print("-" * 60)
    print("结论: " + ("全部通过" if all_ok else "存在失败项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
