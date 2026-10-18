
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"

st.set_page_config(
    page_title="样本增益 AI",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 样本增益 AI")
st.caption("基于大语言模型的少样本文本分类数据增强与效果评估平台")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择展示模块",
    [
        "项目总览",
        "实验结果",
        "消融实验",
        "错误案例分析",
        "增强样本查看",
    ],
)


def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            st.warning(f"读取文件失败：{path.name}（{exc}）")
    return None


def load_json(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            st.warning(f"读取文件失败：{path.name}（{exc}）")
    return None


def show_csv(path, title):
    st.subheader(title)
    df = load_csv(path)
    if df is None:
        st.info(f"暂未找到文件：{path.relative_to(ROOT)}")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"下载{title} CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=path.name,
            mime="text/csv",
        )


if page == "项目总览":
    st.header("项目总览")

    st.markdown(
        """
        本项目研究：在训练样本有限时，利用大语言模型生成合成文本，
        再观察这些样本能否改善文本分类模型的表现。

        **实验流程**

        1. 获取并预处理短信数据集
        2. 每类选择 50 条样本，建立少样本训练集
        3. 训练 TF-IDF + 逻辑回归基线模型
        4. 使用大语言模型生成带标签的合成短信
        5. 比较增强前后的分类效果
        6. 通过消融实验和错误案例分析研究结果
        """
    )

    a, b, c = st.columns(3)
    a.metric("少样本训练集", "100 条")
    b.metric("固定测试集", "1,032 条")
    c.metric("合成样本实验", "6 条 / 10 条")

    st.info(
        "当前实验的增强效果较小，且可能因样本内容而变化。"
        "平台会展示实际实验结果，不将初步结果视为普遍结论。"
    )

    st.subheader("已有实验图表")
    figure_dir = REPORTS / "figures"
    for name in [
        "macro_f1_comparison.png",
        "metrics_comparison.png",
    ]:
        path = figure_dir / name
        if path.exists():
            st.image(str(path), caption=name, use_container_width=True)

    st.subheader("实验报告")
    report_path = REPORTS / "final_experiment_report.md"
    if report_path.exists():
        st.download_button(
            "下载实验报告 Markdown",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/markdown",
        )
    else:
        st.caption("尚未找到 final_experiment_report.md，请先运行报告生成脚本。")


elif page == "实验结果":
    st.header("基线模型与增强模型")

    metrics = load_csv(REPORTS / "compare_metrics.csv")
    if metrics is not None:
        st.dataframe(metrics, use_container_width=True)

        numeric_cols = metrics.select_dtypes(include="number").columns.tolist()
        preferred = [
            c for c in numeric_cols
            if any(key in c.lower() for key in ["accuracy", "f1", "precision", "recall"])
        ]

        if preferred:
            chosen = st.selectbox("选择指标", preferred)
            st.bar_chart(metrics.set_index(metrics.columns[0])[chosen])
        else:
            st.caption("未识别到可直接绘图的指标列，请查看上方原始结果表。")
    else:
        st.warning("找不到 compare_metrics.csv，请先运行对照实验脚本。")

    st.markdown(
        """
        **指标说明**

        - Accuracy：全部样本中预测正确的比例。
        - Macro Precision：分别计算各类别精确率后取平均。
        - Macro Recall：分别计算各类别召回率后取平均。
        - Macro-F1：分别计算各类别 F1 后取平均，更关注类别间的整体表现。
        """
    )


elif page == "消融实验":
    st.header("不同增强样本数量的影响")

    df = load_csv(REPORTS / "ablation_metrics.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        f1_cols = [c for c in numeric_cols if "f1" in c.lower()]

        if f1_cols:
            chosen = st.selectbox("选择展示指标", f1_cols)
            st.line_chart(df.set_index(df.columns[0])[chosen])
        else:
            st.info("未自动识别到 F1 列，请查看上方表格中的实际列名。")
    else:
        st.warning("找不到 ablation_metrics.csv。")


elif page == "错误案例分析":
    st.header("错误案例分析")

    summary = load_csv(
        REPORTS / "error_analysis" / "error_analysis_summary.csv"
    )
    details = load_csv(
        REPORTS / "error_analysis" / "error_analysis_details.csv"
    )

    if summary is not None:
        st.subheader("预测变化统计")
        st.dataframe(summary, use_container_width=True)
        if "change" in summary.columns and "count" in summary.columns:
            st.bar_chart(summary.set_index("change")["count"])

    if details is not None:
        st.subheader("预测变化明细")

        if "change" in details.columns:
            choices = ["全部"] + sorted(
                details["change"].dropna().astype(str).unique().tolist()
            )
            selected = st.selectbox("筛选变化类型", choices)

            if selected != "全部":
                details = details[details["change"] == selected]

        st.dataframe(details, use_container_width=True)

        st.download_button(
            "下载当前筛选结果",
            data=details.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_error_cases.csv",
            mime="text/csv",
        )
    else:
        st.warning("找不到错误案例明细文件。")


elif page == "增强样本查看":
    st.header("大语言模型生成的增强样本")

    candidates = [
        DATA / "augmented_samples_reviewed_completed.csv",
        DATA / "augmented_samples_reviewed.csv",
        DATA / "augmented_samples.csv",
    ]

    sample_path = next((p for p in candidates if p.exists()), None)

    if sample_path is None:
        st.warning("未找到增强样本 CSV，请检查 data/processed 文件夹。")
    else:
        samples = load_csv(sample_path)

        if samples is not None:
            st.caption(f"数据来源：{sample_path.relative_to(ROOT)}")

            if "label" in samples.columns:
                labels = ["全部"] + sorted(
                    samples["label"].dropna().astype(str).unique().tolist()
                )
                selected_label = st.selectbox("按类别筛选", labels)

                if selected_label != "全部":
                    samples = samples[
                        samples["label"].astype(str) == selected_label
                    ]

            st.dataframe(samples, use_container_width=True)

            st.download_button(
                "下载当前样本",
                data=samples.to_csv(index=False).encode("utf-8-sig"),
                file_name="augmented_samples_view.csv",
                mime="text/csv",
            )

st.divider()
st.caption("样本增益 AI | 实验演示版")
