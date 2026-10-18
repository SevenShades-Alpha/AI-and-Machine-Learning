
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import json
import io

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
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

# =========================
# 1. 项目路径与实验配置
# =========================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

for directory in [
    RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLES_PER_CLASS = 50


# =========================
# 2. 下载并读取数据
# =========================

def load_dataset():
    """优先读取本地数据；没有时下载 UCI 官方数据集。"""

    candidates = [
        RAW_DIR / "SMSSpamCollection",
        RAW_DIR / "SMSSpamCollection.tsv",
    ]

    data_file = next(
        (p for p in candidates if p.exists()), None
    )

    if data_file is None:
        print("正在从 UCI 下载数据集……")

        try:
            with urlopen(DATA_URL, timeout=60) as response:
                zip_bytes = response.read()
        except Exception as exc:
            raise RuntimeError(
                "数据下载失败。请检查网络，或从 UCI 官方页面"
                "下载 ZIP，将其中的 SMSSpamCollection 文件"
                f"放入 {RAW_DIR}"
            ) from exc

        with ZipFile(io.BytesIO(zip_bytes)) as archive:
            # 只提取目标数据文件，不解压其他文件
            target = next(
                (
                    name for name in archive.namelist()
                    if Path(name).name == "SMSSpamCollection"
                ),
                None,
            )

            if target is None:
                raise FileNotFoundError(
                    "ZIP 文件中没有找到 SMSSpamCollection"
                )

            content = archive.read(target)

        data_file = RAW_DIR / "SMSSpamCollection"
        data_file.write_bytes(content)

        print(f"数据已保存到：{data_file}")

    df = pd.read_csv(
        data_file,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        on_bad_lines="error",
    )

    df = df.dropna(subset=["label", "text"]).copy()

    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    df = df[
        df["label"].isin(["ham", "spam"])
        & df["text"].ne("")
    ].copy()

    # 删除完全重复的短信，避免相同文本落入不同数据集
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("数据集必须同时包含 ham 和 spam 两个类别")

    print(f"\n清洗后的样本数：{len(df)}")
    print("\n类别分布：")
    print(df["label"].value_counts().to_string())

    return df


# =========================
# 3. 划分训练集与固定测试集
# =========================

def prepare_splits(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # 先划分测试集，再从训练集中抽取少样本
    few_shot_df = (
        train_df.groupby("label", group_keys=False)
        .sample(
            n=min(
                SAMPLES_PER_CLASS,
                int(train_df["label"].value_counts().min()),
            ),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    # 保存数据，供后续增强模块直接复用
    train_df.to_csv(
        PROCESSED_DIR / "train_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    few_shot_df.to_csv(
        PROCESSED_DIR / "train_fewshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n完整训练集样本数：", len(train_df))
    print("少样本训练集样本数：", len(few_shot_df))
    print("固定测试集样本数：", len(test_df))

    return few_shot_df, test_df


# =========================
# 4. 创建基线模型
# =========================

def build_model():
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


# =========================
# 5. 训练、预测与评估
# =========================

def evaluate_model(model, test_df, name):
    X_test = test_df["text"]
    y_true = test_df["label"]
    y_pred = model.predict(X_test)

    metrics = {
        "experiment": name,
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

    print(f"\n===== {name} =====")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\n分类报告：")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["ham", "spam"],
            zero_division=0,
        )
    )

    print("混淆矩阵（行=真实标签，列=预测标签）：")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["ham", "spam"],
        )
    )

    prediction_df = test_df.copy()
    prediction_df["predicted_label"] = y_pred
    prediction_df.to_csv(
        REPORT_DIR / f"{name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


def main():
    df = load_dataset()

    few_shot_df, test_df = prepare_splits(df)

    model = build_model()

    model.fit(
        few_shot_df["text"],
        few_shot_df["label"],
    )

    metrics = evaluate_model(
        model,
        test_df,
        name="baseline",
    )

    model_path = ARTIFACT_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)

    metrics_path = REPORT_DIR / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n基线实验完成！")
    print(f"模型保存位置：{model_path}")
    print(f"指标保存位置：{metrics_path}")
    print(f"测试集保存位置：{PROCESSED_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
