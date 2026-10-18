
import os
from pathlib import Path
from openai import OpenAI

from token_tracker import can_make_request, record_usage

ROOT = Path(__file__).resolve().parents[1]
env_file = ROOT / ".env"

# 读取项目根目录中的 .env 文件
if not env_file.exists():
    raise FileNotFoundError(
        f"没有找到密钥文件：{env_file}"
    )

for line in env_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()

    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'")
        )

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key or api_key == "把真实密钥填在这里":
    raise RuntimeError(
        "没有读取到有效的 DASHSCOPE_API_KEY，请检查 .env 文件"
    )

# 请求前先检查预算
if not can_make_request(reserve_tokens=1000):
    raise RuntimeError(
        "剩余预算不足，已阻止本次 API 调用。"
    )

client = OpenAI(
    api_key=api_key,
    base_url=(
        "https://ws-g8sfxw3exfbuuwna.cn-beijing."
        "maas.aliyuncs.com/compatible-mode/v1"
    ),
    timeout=60.0,
    max_retries=0,
)

MODEL_NAME = "qwen3.7-max"

print("即将调用模型，请注意这会产生一次真实 API 请求。")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": (
                "Write one short, ordinary English SMS message. "
                "Output only the message, with no explanation."
            ),
        }
    ],
    max_tokens=500,
)

content = response.choices[0].message.content or ""
usage = response.usage

print("\n模型回复：")
print(content)

if usage is None:
    print("\n警告：接口未返回 Token 用量。")
    print("请先检查服务商控制台，不要反复重试。")
else:
    record_usage(
        model=MODEL_NAME,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        note="test_api 单条短信测试",
    )

    print("\n本次用量：")
    print(f"输入 Token：{usage.prompt_tokens}")
    print(f"输出 Token：{usage.completion_tokens}")
    print(f"总 Token：{usage.total_tokens}")
