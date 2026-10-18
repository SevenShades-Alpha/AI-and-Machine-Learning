from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
USAGE_FILE = ROOT / "reports" / "token_usage.json"

# 项目内部预算上限：10 万 Token
TOKEN_BUDGET = 100_000


def load_records():
    """读取已有的调用记录。"""
    if not USAGE_FILE.exists():
        return []

    try:
        return json.loads(
            USAGE_FILE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"无法读取 Token 记录文件：{USAGE_FILE}"
        ) from exc


def get_total_tokens():
    """累计已经记录的总 Token 数。"""
    return sum(
        int(record.get("total_tokens", 0))
        for record in load_records()
    )


def can_make_request(reserve_tokens=1000):
    """判断是否还有足够的预算空间。"""
    used = get_total_tokens()
    remaining = TOKEN_BUDGET - used

    print(f"已记录 Token：{used:,}")
    print(f"剩余项目预算：{remaining:,}")

    return remaining >= reserve_tokens


def record_usage(
    model,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    note="",
):
    """记录一次 API 请求的实际 Token 用量。"""
    USAGE_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    records = load_records()

    record = {
        "time": datetime.now().astimezone().isoformat(),
        "model": model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "note": note,
    }

    records.append(record)

    USAGE_FILE.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    used = sum(
        item["total_tokens"] for item in records
    )

    print(f"本次记录：{record['total_tokens']:,} Tokens")
    print(f"累计记录：{used:,} / {TOKEN_BUDGET:,} Tokens")

    if used >= TOKEN_BUDGET:
        print("警告：已达到项目内部 Token 预算上限。")

    return record


if __name__ == "__main__":
    print("Token 预算检查")
    print("-" * 30)
    can_make_request()