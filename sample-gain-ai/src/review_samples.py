
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data" / "processed" / "augmented_samples.csv"
OUTPUT = ROOT / "data" / "processed" / "augmented_samples_reviewed.csv"

if not SOURCE.exists():
    raise FileNotFoundError(f"找不到原始文件：{SOURCE}")

df = pd.read_csv(SOURCE)

# 检查必需列
required = {"text", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV 必须包含列：{required}")

# 保留原始内容，增加审核字段
df["review_status"] = "pending"
df["reviewed_label"] = ""
df["review_notes"] = ""

# 输出一份新的审核文件，不覆盖原始数据
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("审核版数据集已生成！")
print(f"文件位置：{OUTPUT}")
print(f"样本总数：{len(df)}")
print("\n标签数量：")
print(df["label"].value_counts().to_string())
print("\n请打开 augmented_samples_reviewed.csv，逐条审核。")
