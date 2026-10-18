
import os
import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from token_tracker import can_make_request, record_usage

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "train_fewshot.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "augmented_samples.csv"
RAW_DIR = ROOT / "reports" / "augmentation_raw"

MODEL_NAME = "qwen3.7-max"
LABELS = ["ham", "spam"]
EXAMPLES_PER_CLASS = 5
GENERATE_PER_CLASS = 5
MAX_OUTPUT_TOKENS = 1200


def load_api_key():
    """读取项目根目录中的 .env。"""
    env_file = ROOT / ".env"

    if not env_file.exists():
        raise FileNotFoundError(f"没有找到密钥文件：{env_file}")

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
        raise RuntimeError("请检查 .env 中的 DASHSCOPE_API_KEY。")

    return api_key


def load_training_data():
    """只读取少样本训练集。"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到训练集：{DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"训练集需要包含 {required} 两列，"
            f"当前列名为：{list(df.columns)}"
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)]
    df = df[df["text"] != ""]

    return df


def build_prompt(label, examples):
    """构造单一类别的数据增强提示词。"""
    examples_text = "\n".join(
        f"{i + 1}. {text}"
        for i, text in enumerate(examples)
    )

    if label == "ham":
        definition = (
            "Normal, legitimate personal or service messages. "
            "They are not unsolicited spam."
        )
    else:
        definition = (
            "Unsolicited spam messages, such as deceptive promotions, "
            "unwanted commercial offers, or suspicious prize claims."
        )

    return f"""
You are creating synthetic SMS data for a text classification experiment.

Target label: {label}
Label definition: {definition}

Examples from the training set:
{examples_text}

Generate exactly {GENERATE_PER_CLASS} NEW and diverse English SMS messages
with the target label. Do not copy the examples. Keep messages realistic
and concise. Do not include real personal information.

Return only a valid JSON array. Each item must have this format:
{{"text": "message text", "label": "{label}"}}

Do not include markdown fences or explanations.
""".strip()


def parse_generated_json(content, expected_label):
    """解析模型返回的 JSON，并检查基础格式。"""
    content = content.strip()

    # 兼容模型偶尔添加的 Markdown 代码围栏
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("[")
    end = content.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("模型返回内容中没有找到 JSON 数组。")

    data = json.loads(content[start:end + 1])

    if not isinstance(data, list):
        raise ValueError("模型返回的结果不是 JSON 数组。")

    results = []

    for item in data:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip().lower()

        if not text:
            continue

        # 每次请求只允许生成指定类别，避免标签混入
        if label != expected_label:
            continue

        results.append({
            "text": text,
            "label": expected_label,
            "source": "llm",
            "review_status": "pending",
        })

    return results


def save_results(rows):
    """每次生成后立即保存已有结果。"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(
        rows,
        columns=["text", "label", "source", "review_status"]
    )

    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE)
        new_df = pd.concat([old_df, new_df], ignore_index=True)

    # 去除完全相同的文本，避免重复样本
    new_df["text"] = new_df["text"].astype(str).str.strip()
    new_df = new_df.drop_duplicates(subset=["text"], keep="first")

    new_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()

    print("训练集读取成功。")
    print(f"训练集样本数：{len(df)}")
    print("各标签样本数：")
    print(df["label"].value_counts().to_string())
    print(f"\n本轮目标：每类生成 {GENERATE_PER_CLASS} 条。")

    api_key = load_api_key()

    client = OpenAI(
        api_key=api_key,
        base_url=(
            "https://ws-g8sfxw3exfbuuwna.cn-beijing."
            "maas.aliyuncs.com/compatible-mode/v1"
        ),
        timeout=60.0,
        max_retries=0,
    )

    all_results = []
    used_texts = set(df["text"].str.lower())

    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE)
        if "text" in old_df.columns:
            used_texts.update(
                old_df["text"].dropna().astype(str).str.lower()
            )

    for label in LABELS:
        examples = (
            df[df["label"] == label]["text"]
            .drop_duplicates()
            .head(EXAMPLES_PER_CLASS)
            .tolist()
        )

        if len(examples) < EXAMPLES_PER_CLASS:
            print(f"\n{label} 类样本不足，跳过该类别。")
            continue

        # 请求前检查预算。每个类别单独请求，避免一次生成过多。
        if not can_make_request(reserve_tokens=2000):
            print("预算余量不足，停止后续生成。")
            break

        prompt = build_prompt(label, examples)
        print(f"\n正在生成 {label} 类数据……")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            print(f"API 请求失败：{type(exc).__name__}: {exc}")
            print("为避免重复计费，本程序不会自动重试。")
            print("请先检查服务商控制台用量，再决定是否继续。")
            break

        content = response.choices[0].message.content or ""

        # 无论 JSON 能否解析，都先保存原始响应，方便排查。
        raw_file = RAW_DIR / f"{label}_{int(time.time())}.txt"
        raw_file.write_text(content, encoding="utf-8")

        usage = response.usage

        if usage is None:
            print("接口没有返回 Token 用量，暂停后续生成。")
            print("请先检查服务商控制台。")
            break

        record_usage(
            model=MODEL_NAME,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            note=f"augmentation 小样本生成：{label}",
        )

        try:
            rows = parse_generated_json(content, label)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"JSON 解析失败：{exc}")
            print(f"原始结果已保存到：{raw_file}")
            continue

        # 过滤训练集及历史增强结果中已有的文本
        fresh_rows = []

        for row in rows:
            normalized = row["text"].lower()

            if normalized in used_texts:
                continue

            used_texts.add(normalized)
            fresh_rows.append(row)

        all_results.extend(fresh_rows)

        # 每个类别处理后立即保存，降低中断导致的数据丢失风险。
        save_results(all_results)

        print(f"模型返回有效样本：{len(rows)}")
        print(f"去重后新增样本：{len(fresh_rows)}")
        print(f"原始响应已保存：{raw_file}")
        print(f"当前结果文件：{OUTPUT_FILE}")

    print("\n本轮生成流程结束。")
    print(f"本轮新增样本数：{len(all_results)}")
    print(f"结果文件：{OUTPUT_FILE}")
    print("所有生成样本均标记为 pending，尚未通过人工质量审核。")


if __name__ == "__main__":
    main()
