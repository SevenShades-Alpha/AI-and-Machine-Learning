
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"

FEWSHOT_FILE = DATA_DIR / "train_fewshot.csv"
TEST_FILE = DATA_DIR / "test.csv"
AUGMENTED_FILE = DATA_DIR / "augmented_samples.csv"

RANDOM_STATE = 42
LABELS = ["ham", "spam"]

# 第一轮仅作探索性实验，允许使用待审核样本。
# 正式实验前应审核样本，并将此项改为 False，
# 同时将合格样本的 review_status 标记为 approved。
INCLUDE_PENDING = True


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} 必须包含 text 和 label 列，"
            f"当前列名：{list(df.columns)}"
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[
        df["text"].ne("")
        & df["label"].isin(LABELS)
    ].copy()

    return df.reset_index(drop=True)


def build_model():
    """与 baseline.py 使用相同的模型参数。"""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate(model, test_df, experiment_name):
    y_true = test_df["label"]
    y_pred = model.predict(test_df["text"])

    metrics = {
        "experiment": experiment_name,
        "test_samples": int(len(test_df)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    print(f"\n===== {experiment_name} =====")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\n分类报告：")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    ))

    print("混淆矩阵（行=真实标签，列=预测标签）：")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    ))

    prediction_df = test_df.copy()
    prediction_df["predicted_label"] = y_pred
    prediction_df["correct"] = (
        prediction_df["label"] == prediction_df["predicted_label"]
    )

    prediction_df.to_csv(
        REPORT_DIR / f"{experiment_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics, prediction_df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 只读取已有文件，不重新划分数据集
    train_df = load_data(FEWSHOT_FILE)
    test_df = load_data(TEST_FILE)

    print(f"少样本训练集：{len(train_df)} 条")
    print(f"固定测试集：{len(test_df)} 条")

    # 保留原始训练集，不让合成数据覆盖它
    augmented_df = pd.read_csv(
        AUGMENTED_FILE,
        encoding="utf-8-sig",
    )

    required = {"text", "label"}
    if not required.issubset(augmented_df.columns):
        raise ValueError("合成数据必须包含 text 和 label 列。")

    if not INCLUDE_PENDING:
        if "review_status" not in augmented_df.columns:
            raise ValueError("合成数据缺少 review_status 列。")

        augmented_df = augmented_df[
            augmented_df["review_status"].astype(str).str.lower()
            == "approved"
        ].copy()
    else:
        print(
            "\n注意：当前使用待审核合成样本，"
            "实验结果仅用于初步探索。"
        )

    augmented_df = augmented_df.dropna(
        subset=["text", "label"]
    ).copy()

    augmented_df["text"] = (
        augmented_df["text"].astype(str).str.strip()
    )
    augmented_df["label"] = (
        augmented_df["label"].astype(str).str.strip().str.lower()
    )

    augmented_df = augmented_df[
        augmented_df["text"].ne("")
        & augmented_df["label"].isin(LABELS)
    ].copy()

    # 防止合成文本与训练集或测试集存在完全相同的文本
    train_texts = set(train_df["text"].str.casefold())
    test_texts = set(test_df["text"].str.casefold())

    before = len(augmented_df)

    augmented_df = augmented_df[
        ~augmented_df["text"].str.casefold().isin(train_texts)
        & ~augmented_df["text"].str.casefold().isin(test_texts)
    ].copy()

    # 合成数据内部去重
    augmented_df = augmented_df.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    print(f"可用合成样本：{len(augmented_df)} 条")
    print(f"过滤重复或测试集重合文本：{before - len(augmented_df)} 条")

    print("\n合成数据类别分布：")
    if len(augmented_df):
        print(augmented_df["label"].value_counts().to_string())
    else:
        print("没有可用合成样本，无法进行增强实验。")
        return

    # 实验 A：原始少样本基线
    baseline_model = build_model()
    baseline_model.fit(train_df["text"], train_df["label"])

    baseline_metrics, baseline_predictions = evaluate(
        baseline_model,
        test_df,
        "compare_baseline",
    )

    # 实验 B：少样本训练数据 + 合成数据
    combined_df = pd.concat(
        [
            train_df[["text", "label"]],
            augmented_df[["text", "label"]],
        ],
        ignore_index=True,
    )

    augmented_model = build_model()
    augmented_model.fit(
        combined_df["text"],
        combined_df["label"],
    )

    augmented_metrics, augmented_predictions = evaluate(
        augmented_model,
        test_df,
        "compare_augmented",
    )

    # 保存独立模型，不覆盖原 baseline_model.joblib
    joblib.dump(
        baseline_model,
        ARTIFACT_DIR / "compare_baseline_model.joblib",
    )
    joblib.dump(
        augmented_model,
        ARTIFACT_DIR / "compare_augmented_model.joblib",
    )

    # 汇总指标及变化
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    summary = {
        "experiment_note": (
            "exploratory_pending_samples"
            if INCLUDE_PENDING
            else "reviewed_samples_only"
        ),
        "training_samples_baseline": int(len(train_df)),
        "training_samples_augmented": int(len(combined_df)),
        "augmentation_samples_used": int(len(augmented_df)),
        "test_samples": int(len(test_df)),
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "delta_augmented_minus_baseline": {
            name: round(
                augmented_metrics[name] - baseline_metrics[name],
                6,
            )
            for name in metric_names
        },
    }

    summary_path = REPORT_DIR / "compare_metrics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison = pd.DataFrame([
        {
            "experiment": "baseline",
            **{
                name: baseline_metrics[name]
                for name in metric_names
            },
        },
        {
            "experiment": "augmented",
            **{
                name: augmented_metrics[name]
                for name in metric_names
            },
        },
    ])

    comparison.to_csv(
        REPORT_DIR / "compare_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 保存增强模型新增纠错和新增误判的样本
    error_comparison = test_df[["text", "label"]].copy()
    error_comparison["baseline_prediction"] = (
        baseline_predictions["predicted_label"].values
    )
    error_comparison["augmented_prediction"] = (
        augmented_predictions["predicted_label"].values
    )

    error_comparison["change"] = "both_same"
    error_comparison.loc[
        (error_comparison["baseline_prediction"] != error_comparison["label"])
        & (error_comparison["augmented_prediction"] == error_comparison["label"]),
        "change",
    ] = "corrected"

    error_comparison.loc[
        (error_comparison["baseline_prediction"] == error_comparison["label"])
        & (error_comparison["augmented_prediction"] != error_comparison["label"]),
        "change",
    ] = "new_error"

    error_comparison.to_csv(
        REPORT_DIR / "compare_error_changes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 指标变化（增强模型 - 基线模型）=====")
    for name in metric_names:
        delta = (
            augmented_metrics[name] - baseline_metrics[name]
        )
        print(f"{name}: {delta:+.4f}")

    print("\n对照实验完成。")
    print(f"指标汇总：{summary_path}")
    print(f"指标表格：{REPORT_DIR / 'compare_metrics.csv'}")
    print(f"错误变化：{REPORT_DIR / 'compare_error_changes.csv'}")
    print("原始 baseline.py 及其模型文件未被覆盖。")


if __name__ == "__main__":
    main()
