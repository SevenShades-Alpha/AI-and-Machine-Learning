
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 已有实验结果文件
changes_path = ROOT / "reports" / "compare_error_changes.csv"
baseline_path = ROOT / "reports" / "compare_baseline_predictions.csv"
augmented_path = ROOT / "reports" / "compare_augmented_predictions.csv"

# 输出目录
output_dir = ROOT / "reports" / "error_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# 优先读取已经生成的错误变化表
if changes_path.exists():
    df = pd.read_csv(changes_path)

    required = {
        "text",
        "label",
        "baseline_prediction",
        "augmented_prediction",
        "change",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"compare_error_changes.csv 缺少列：{sorted(missing)}；"
            f"当前列名为：{df.columns.tolist()}"
        )

else:
    # 如果变化表不存在，就读取两份预测结果并按文本与真实标签匹配
    if not baseline_path.exists() or not augmented_path.exists():
        raise FileNotFoundError(
            "找不到实验结果文件。请检查 reports 文件夹。"
        )

    base = pd.read_csv(baseline_path)
    aug = pd.read_csv(augmented_path)

    def find_column(df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    base_text = find_column(base, ["text"])
    aug_text = find_column(aug, ["text"])
    base_label = find_column(base, ["label", "true_label"])
    aug_label = find_column(aug, ["label", "true_label"])
    base_pred = find_column(
        base, ["baseline_prediction", "prediction", "predicted_label"]
    )
    aug_pred = find_column(
        aug, ["augmented_prediction", "prediction", "predicted_label"]
    )

    if None in [
        base_text, aug_text, base_label, aug_label, base_pred, aug_pred
    ]:
        raise ValueError(
            "无法识别预测文件中的列名。\n"
            f"基线文件列名：{base.columns.tolist()}\n"
            f"增强文件列名：{aug.columns.tolist()}"
        )

    base = base[[base_text, base_label, base_pred]].rename(
        columns={
            base_text: "text",
            base_label: "label",
            base_pred: "baseline_prediction",
        }
    )

    aug = aug[[aug_text, aug_label, aug_pred]].rename(
        columns={
            aug_text: "text",
            aug_label: "label",
            aug_pred: "augmented_prediction",
        }
    )

    # 按文本和真实标签匹配，避免直接按行号错配
    df = base.merge(
        aug,
        on=["text", "label"],
        how="inner",
        validate="one_to_one",
    )

    df["change"] = "both_same"
    df.loc[
        (df["baseline_prediction"] != df["label"])
        & (df["augmented_prediction"] == df["label"]),
        "change",
    ] = "corrected"

    df.loc[
        (df["baseline_prediction"] == df["label"])
        & (df["augmented_prediction"] != df["label"]),
        "change",
    ] = "new_error"

# 统一保存明细
details_path = output_dir / "error_analysis_details.csv"
df.to_csv(details_path, index=False, encoding="utf-8-sig")

# 统计变化类型
summary = (
    df["change"]
    .value_counts()
    .rename_axis("change")
    .reset_index(name="count")
)

summary_path = output_dir / "error_analysis_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

# 绘制预测变化图
plt.figure(figsize=(8, 5))
plt.bar(summary["change"], summary["count"])
plt.title("Prediction Changes: Baseline vs Augmented")
plt.xlabel("Change Type")
plt.ylabel("Number of Samples")
plt.tight_layout()

figure_path = output_dir / "prediction_changes.png"
plt.savefig(figure_path, dpi=200)
plt.close()

print("错误案例分析完成！")
print(f"参与分析的样本数：{len(df)}")
print("\n变化统计：")
print(summary.to_string(index=False))
print(f"\n明细文件：{details_path}")
print(f"统计文件：{summary_path}")
print(f"变化图：{figure_path}")
