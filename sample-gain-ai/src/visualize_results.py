
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = REPORT_DIR / "ablation_metrics.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"找不到实验结果：{csv_path}\n"
        "请先确认 ablation_experiment.py 已经运行成功。"
    )

df = pd.read_csv(csv_path)

required = {"experiment", "accuracy", "f1_macro"}
if not required.issubset(df.columns):
    raise ValueError(
        f"结果文件缺少必要列：{required - set(df.columns)}"
    )

# 根据实际增强样本数量设置显示名称
names = []
for _, row in df.iterrows():
    count = int(row["augmented_samples_requested"])
    if count == 0:
        names.append("Baseline (0)")
    else:
        names.append(f"Augmented ({count})")

# 图1：不同增强样本数量对应的 Macro-F1
plt.figure(figsize=(8, 5))
bars = plt.bar(names, df["f1_macro"])
plt.title("Effect of Augmented Sample Count on Macro-F1")
plt.xlabel("Experiment")
plt.ylabel("Macro-F1")
plt.ylim(0, 1)
plt.bar_label(bars, fmt="%.4f", padding=3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "macro_f1_comparison.png",
    dpi=200
)
plt.close()

# 图2：Accuracy 和 Macro-F1 对比
x = range(len(df))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(
    [i - width / 2 for i in x],
    df["accuracy"],
    width,
    label="Accuracy"
)
plt.bar(
    [i + width / 2 for i in x],
    df["f1_macro"],
    width,
    label="Macro-F1"
)
plt.xticks(list(x), names)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "metrics_comparison.png",
    dpi=200
)
plt.close()

print("图表生成成功！")
print(f"保存目录：{OUTPUT_DIR}")
print("1. macro_f1_comparison.png")
print("2. metrics_comparison.png")
print("\n实验结果：")
print(df[["experiment", "accuracy", "f1_macro"]].to_string(index=False))
