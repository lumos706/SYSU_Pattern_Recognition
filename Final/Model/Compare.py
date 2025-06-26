import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings

warnings.filterwarnings("ignore")
# 设置中文显示和字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 12

def setup_logger(log_file):
    """配置同时输出到文件和终端的日志系统"""
    logger = logging.getLogger('LogisticRegression')
    logger.setLevel(logging.INFO)

    # 清除之前的处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建输出目录
    output_dir = os.path.dirname(log_file)
    os.makedirs(output_dir, exist_ok=True)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def compare_models():
    """
    比较所有模型性能，生成对比图表和报告

    功能：
    1. 从各模型目录读取评估指标
    2. 创建综合性能对比表
    3. 生成多种可视化图表
    4. 创建详细性能对比报告
    """
    # 设置日志
    logger = setup_logger('../outputs/comparison/model_comparison.log')
    logger.info("===== 开始模型性能对比分析 =====")

    # 确保输出目录存在
    output_dir = '../outputs/comparison'
    os.makedirs(output_dir, exist_ok=True)

    # 模型列表和对应的显示名称
    models = {
        "RandomForest": "随机森林",
        "LogisticRegression": "逻辑回归",
        "KMeans": "K-Means",
        "SVM": "支持向量机"
    }

    # 存储所有模型的性能指标
    all_metrics = []

    logger.info("收集各模型性能指标...")
    for model_key, model_name in models.items():
        metrics_path = os.path.join('../outputs', model_key, 'metrics.csv')

        if os.path.exists(metrics_path):
            try:
                # 读取指标数据
                df = pd.read_csv(metrics_path)
                df['Model'] = model_name  # 添加模型名称列
                all_metrics.append(df)
                logger.info(f"已加载 {model_name} 的性能指标")
            except Exception as e:
                logger.error(f"加载 {model_name} 指标失败: {str(e)}")
        else:
            logger.warning(f"找不到 {model_name} 的性能指标文件: {metrics_path}")

    if not all_metrics:
        logger.error("未找到任何模型性能数据，对比分析终止")
        return

    # 合并所有模型的性能数据
    comparison_df = pd.concat(all_metrics, ignore_index=True)

    # 保存对比结果到CSV
    comparison_csv_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"模型对比数据已保存至: {comparison_csv_path}")

    # 创建可视化图表目录
    plots_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. 综合性能对比图
    logger.info("生成综合性能对比图...")
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=comparison_df,
                palette='viridis', edgecolor='black')

    plt.title('不同分类模型性能对比', fontsize=16, fontweight='bold')
    plt.ylabel('分数', fontsize=14)
    plt.xlabel('评估指标', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(title='模型', loc='best', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数据标签
    for p in plt.gca().patches:
        height = p.get_height()
        if not np.isnan(height):  # 确保高度不是NaN
            plt.gca().annotate(
                f'{height:.3f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=10
            )

    comparison_plot_path = os.path.join(plots_dir, 'overall_comparison.png')
    plt.savefig(comparison_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"综合性能对比图已保存至: {comparison_plot_path}")

    # 2. 各指标详细对比图
    logger.info("生成各指标详细对比图...")
    metrics = comparison_df['Metric'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        # 筛选当前指标的数据
        metric_df = comparison_df[comparison_df['Metric'] == metric]

        # 按模型名称排序以保持一致性
        model_order = ["随机森林", "逻辑回归", "K-Means", "支持向量机"]
        metric_df['Model'] = pd.Categorical(metric_df['Model'], categories=model_order, ordered=True)
        metric_df = metric_df.sort_values('Model')

        ax = axes[i]
        sns.barplot(x='Model', y='Value', data=metric_df,
                    ax=ax, palette='coolwarm', edgecolor='black')

        ax.set_title(f'{metric} 对比', fontsize=14)
        ax.set_ylabel('分数')
        ax.set_xlabel('')
        ax.set_ylim(0, 1.05)

        # 添加数据标签
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:.3f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=10
            )

    # 移除多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    metrics_comparison_path = os.path.join(plots_dir, 'metrics_comparison.png')
    plt.savefig(metrics_comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"各指标详细对比图已保存至: {metrics_comparison_path}")

    # 4. 创建性能对比报告
    logger.info("生成性能对比报告...")
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("学业困难学生识别模型性能对比报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 汇总表格
        f.write("各模型性能指标汇总:\n")
        f.write("-" * 60 + "\n")

        # 重塑数据以便于阅读
        pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Value')
        f.write(pivot_df.to_string(float_format="%.4f"))
        f.write("\n\n")

        # 找出每个指标的最佳模型
        f.write("各指标最佳模型:\n")
        f.write("-" * 60 + "\n")

        best_models = {}
        for metric in metrics:
            best_idx = pivot_df[metric].idxmax()
            best_value = pivot_df.loc[best_idx, metric]
            f.write(f"{metric}: {best_idx} ({best_value:.4f})\n")
            best_models[metric] = best_idx

        f.write("\n")

        # 综合评估
        f.write("综合评估:\n")
        f.write("-" * 60 + "\n")

        # 计算综合得分（简单平均）
        pivot_df['综合得分'] = pivot_df.mean(axis=1)
        overall_best = pivot_df['综合得分'].idxmax()
        overall_score = pivot_df.loc[overall_best, '综合得分']

        f.write(f"综合表现最佳模型: {overall_best} (综合得分: {overall_score:.4f})\n\n")


    logger.info(f"性能对比报告已保存至: {report_path}")

    # 最终总结
    logger.info("===== 模型性能对比分析完成 =====")
    print(f"模型对比分析完成! 结果保存在: {output_dir}")
    print(f"详细报告: {report_path}")
    print(f"可视化图表: {plots_dir}")


if __name__ == "__main__":
    compare_models()