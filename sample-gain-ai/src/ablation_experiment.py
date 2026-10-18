
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "train_fewshot.csv"
test_path = DATA_DIR / "test.csv"
aug_path = DATA_DIR / "augmented_samples.csv"

# 检查文件是否存在
for path in [train_path, test_path, aug_path]:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
aug = pd.read_csv(aug_path)

# 只使用待审核的 AI 合成数据进行探索性实验
# 每组按 ham、spam 分别取相同数量，保持类别平衡
aug = aug.drop_duplicates(subset=["text"]).copy()
aug["label"] = aug["label"].astype(str).str.lower().str.strip()
aug["text"] = aug["text"].astype(str).str.strip()

results = []

for n_per_class in [0, 3, 5]:
    if n_per_class == 0:
        selected = aug.iloc[0:0].copy()
    else:
        selected_parts = []

        for label in ["ham", "spam"]:
            class_samples = aug[aug["label"] == label]

            if len(class_samples) < n_per_class:
                raise ValueError(
                    f"{label} 类只有 {len(class_samples)} 条，"
                    f"不足 {n_per_class} 条。"
                )

            selected_parts.append(
                class_samples.head(n_per_class)
            )

        selected = pd.concat(selected_parts, ignore_index=True)

    # 合并原始少样本训练集和本组的合成样本
    combined = pd.concat(
        [
            train[["label", "text"]],
            selected[["label", "text"]],
        ],
        ignore_index=True,
    )

    # 去除空文本，并排除与原训练文本重复的合成文本
    combined["text"] = combined["text"].astype(str).str.strip()
    combined["label"] = combined["label"].astype(str).str.lower().str.strip()
    combined = combined[combined["text"] != ""]
    combined = combined.drop_duplicates(subset=["text"])

    # 每组使用相同的模型参数
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(combined["text"], combined["label"])

    predictions = model.predict(test["text"])
    y_true = test["label"].astype(str).str.lower().str.strip()

    metrics = {
        "experiment": f"aug_{n_per_class * 2}_samples",
        "original_train_samples": len(train),
        "augmented_samples_requested": len(selected),
        "actual_train_samples": len(combined),
        "test_samples": len(test),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=["ham", "spam"]
        ).tolist(),
    }

    results.append(metrics)

    print(f"\n完成实验：增加 {len(selected)} 条合成样本")
    print(f"训练样本数：{len(combined)}")
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Macro-F1：{metrics['f1_macro']:.4f}")
    print(f"混淆矩阵：{metrics['confusion_matrix']}")

# 保存结果，不覆盖之前的对照实验文件
json_path = REPORT_DIR / "ablation_metrics.json"
csv_path = REPORT_DIR / "ablation_metrics.csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

pd.DataFrame([
    {
        key: value
        for key, value in item.items()
        if key != "confusion_matrix"
    }
    for item in results
]).to_csv(csv_path, index=False, encoding="utf-8-sig")

print("\n全部完成！")
print(f"结果文件：{json_path}")
print(f"结果表格：{csv_path}")
