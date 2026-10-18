
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "final_experiment_report.md"


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def fmt(value, digits=4):
    if value is None:
        return "暂无"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    baseline = read_json(REPORTS / "baseline_metrics.json")
    compare = read_json(REPORTS / "compare_metrics.json")
    ablation = read_csv(REPORTS / "ablation_metrics.csv")
    errors = read_csv(
        REPORTS / "error_analysis" / "error_analysis_summary.csv"
    )

    lines = [
        "# 样本增益 AI：基于大语言模型的少样本文本分类数据增强与效果评估平台",
        "",
        "## 摘要",
        "",
        "本项目研究在少样本文本分类场景中，利用大语言模型生成合成训练样本，"
        "并通过基线对照、增强数据实验、消融实验和错误案例分析评估其效果。",
        "实验使用 SMS 垃圾短信分类数据集，采用 TF-IDF 与逻辑回归作为分类器。",
        "当前结果属于初步实验，仍需通过多随机种子和更多增强样本验证稳定性。",
        "",
        "## 1. 研究目标",
        "",
        "- 构建少样本文本分类基线模型。",
        "- 使用大语言模型生成带标签的合成训练样本。",
        "- 比较增强前后模型在相同测试集上的表现。",
        "- 研究不同增强样本数量对分类效果的影响。",
        "- 分析预测变化与错误案例。",
        "",
        "## 2. 数据与方法",
        "",
        "- 数据集：UCI SMS Spam Collection。",
        "- 标签：ham（正常短信）与 spam（垃圾短信）。",
        "- 少样本训练集：100 条，每类 50 条。",
        "- 测试集：1,032 条。",
        "- 特征方法：TF-IDF，使用一元与二元词组。",
        "- 分类器：逻辑回归，设置 max_iter=1000、class_weight='balanced'。",
        "- 增强方法：调用大语言模型生成带标签的合成短信。",
        "- 评估指标：Accuracy、Macro Precision、Macro Recall、Macro-F1。",
        "",
        "## 3. 基线与增强实验",
        "",
    ]

    if baseline:
        lines += [
            "### 3.1 基线结果",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| Accuracy | {fmt(baseline.get('accuracy'))} |",
            f"| Macro Precision | {fmt(baseline.get('precision_macro'))} |",
            f"| Macro Recall | {fmt(baseline.get('recall_macro'))} |",
            f"| Macro-F1 | {fmt(baseline.get('f1_macro'))} |",
            "",
        ]
    else:
        lines += ["未找到基线指标文件，需补充实验结果。", ""]

    if compare:
        acc_delta = (
            compare.get("accuracy", 0) - baseline.get("accuracy", 0)
            if baseline else None
        )
        f1_delta = (
            compare.get("f1_macro", 0) - baseline.get("f1_macro", 0)
            if baseline else None
        )
        lines += [
            "### 3.2 增强对照结果",
            "",
            "| 指标 | 基线 | 增强后 | 变化 |",
            "|---|---:|---:|---:|",
            f"| Accuracy | {fmt(baseline.get('accuracy') if baseline else None)} "
            f"| {fmt(compare.get('accuracy'))} "
            f"| {fmt(acc_delta) if acc_delta is not None else '暂无'} |",
            f"| Macro-F1 | {fmt(baseline.get('f1_macro') if baseline else None)} "
            f"| {fmt(compare.get('f1_macro'))} "
            f"| {fmt(f1_delta) if f1_delta is not None else '暂无'} |",
            "",
            "本次增强实验使用 10 条合成样本。结果显示指标有小幅变化，"
            "尚不能据此认定提升具有统计显著性或普遍性。",
            "",
        ]
    else:
        lines += ["未找到增强对照指标文件，需补充实验结果。", ""]

    if ablation is not None:
        lines += [
            "## 4. 消融实验",
            "",
            "下表为不同增强样本数量的实验结果。具体列名以 CSV 文件为准。",
            "",
            ablation.to_markdown(index=False),
            "",
            "在当前实验设置下，6 条增强样本方案表现较好；"
            "该结果只适用于本次数据、模型与测试集，仍需进一步验证。",
            "",
        ]
    else:
        lines += ["## 4. 消融实验", "", "未找到消融实验结果文件。", ""]

    if errors is not None:
        lines += [
            "## 5. 错误案例分析",
            "",
            errors.to_markdown(index=False),
            "",
            "错误分析显示，增强样本可能纠正部分原有错误，也可能引入新错误。"
            "应同时关注纠正数量、新增错误及其对应的真实标签。",
            "",
        ]
    else:
        lines += [
            "## 5. 错误案例分析",
            "",
            "未找到错误分析汇总文件。",
            "",
        ]

    lines += [
        "## 6. 局限性",
        "",
        "1. 当前实验使用单一固定测试集，尚未进行多随机种子重复实验。",
        "2. 合成样本数量较少，实验结果可能受具体样本内容影响。",
        "3. 合成样本经过人工初审，但尚缺少独立标注者的一致性验证。",
        "4. 当前仅使用一种传统分类器，尚未比较其他分类方法。",
        "5. 指标变化较小，不能据此宣称数据增强必然有效。",
        "",
        "## 7. 后续计划",
        "",
        "- 在多个随机种子下重复实验，报告均值与标准差。",
        "- 增加不同增强数量和增强策略的比较。",
        "- 进一步分析新增错误与纠正错误的文本特征。",
        "- 增加交互式实验展示界面。",
        "- 完整记录模型参数、数据版本和实验配置。",
        "",
        "## 8. 结论",
        "",
        "本项目完成了少样本基线训练、大语言模型合成数据生成、"
        "增强对照实验、样本数量消融、结果可视化和错误案例分析。",
        "初步结果表明，少量合成数据在当前实验设置下可能改善分类表现，"
        "但效果受样本数量和内容影响，仍需要更系统的实验验证。",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print("实验报告生成成功！")
    print(f"文件位置：{OUTPUT}")


if __name__ == "__main__":
    main()
