
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / "reports" / "baseline_predictions.csv"

df = pd.read_csv(file_path)

errors = df[df["label"] != df["predicted_label"]]

print("测试样本总数：", len(df))
print("预测错误数量：", len(errors))
print("\n错误类型统计：")
print(
    pd.crosstab(
        errors["label"],
        errors["predicted_label"]
    )
)

print("\n错误样本示例（最多展示10条）：")
print(
    errors[["label", "text", "predicted_label"]]
    .head(10)
    .to_string(index=False)
)
