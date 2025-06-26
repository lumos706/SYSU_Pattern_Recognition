import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 设置全局字体
plt.rc("font", family='Microsoft YaHei', weight="bold")


# 配置日志系统
def setup_logger(log_file):
    """配置同时输出到文件和终端的日志系统"""
    logger = logging.getLogger('KMeansClustering')
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


# KMeans包装器类，用于将聚类结果转换为分类预测
class KMeansClassifier:
    def __init__(self, kmeans_model, cluster_to_class_mapping):
        self.kmeans = kmeans_model
        self.mapping = cluster_to_class_mapping

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return np.array([self.mapping[c] for c in clusters])

    def fit(self, X, y=None):
        # KMeans不需要在fit时使用标签
        return self


# 评估函数
def evaluate_model(model, X_test, y_test, logger, output_dir):
    """全面评估模型性能"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 由于KMeans没有概率输出，ROC AUC无法计算
    roc_auc = None

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常学生', '学业困难学生'],
                yticklabels=['正常学生', '学业困难学生'])
    plt.title('KMeans聚类 - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=['正常学生', '学业困难学生'])

    # 记录结果
    logger.info(f"评估完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info("\n==================== 模型性能指标 ====================")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数 (F1-Score): {f1:.4f}")
    logger.info("ROC AUC分数: 未计算（KMeans无概率输出）")

    logger.info("\n==================== 混淆矩阵 ====================")
    logger.info(f"\n{cm}")

    logger.info("\n==================== 分类报告 ====================")
    logger.info(f"\n{report}")

    # 保存评估结果到文件
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"评估指标已保存至: {metrics_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confusion_matrix_path': cm_path
    }


# 聚类特征分析
def analyze_clusters(model, feature_names, X_train, y_train, logger, output_dir):
    """分析聚类特征和重要性"""
    analysis_dir = os.path.join(output_dir, 'cluster_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # 获取聚类中心和标签
    centroids = model.kmeans.cluster_centers_
    n_clusters = centroids.shape[0]

    # 计算每个特征的聚类间差异
    feature_importance = np.std(centroids, axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1]

    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in sorted_idx],
        'Importance': feature_importance[sorted_idx]
    })

    # 保存特征重要性
    importance_path = os.path.join(analysis_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)

    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1], color='skyblue')
    plt.xlabel('特征重要性（聚类间标准差）')
    plt.title('KMeans聚类 - Top 20重要特征')
    plt.tight_layout()
    importance_plot_path = os.path.join(analysis_dir, 'feature_importance.png')
    plt.savefig(importance_plot_path, bbox_inches='tight')
    plt.close()

    # 分析每个聚类的特征分布
    cluster_labels = model.kmeans.labels_

    # 创建聚类特征分析数据框
    cluster_df = pd.DataFrame(X_train, columns=feature_names)
    cluster_df['Cluster'] = cluster_labels
    cluster_df['Actual_Class'] = y_train.values

    # 计算每个聚类的特征均值
    cluster_means = cluster_df.groupby('Cluster').mean()

    # 计算每个聚类中困难学生的比例
    cluster_risk = cluster_df.groupby('Cluster')['Actual_Class'].mean()

    # 保存聚类分析结果
    cluster_means_path = os.path.join(analysis_dir, 'cluster_means.csv')
    cluster_means.to_csv(cluster_means_path)

    # 绘制聚类中心热力图
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('KMeans聚类 - 各聚类特征均值')
    plt.xlabel('聚类编号')
    plt.ylabel('特征')
    heatmap_path = os.path.join(analysis_dir, 'cluster_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    # 记录分析结果
    logger.info("\n==================== 聚类分析结果 ====================")
    logger.info(f"各聚类中学业困难学生比例:\n{cluster_risk}")
    logger.info(f"Top 10重要特征:\n{importance_df.head(10)}")

    return {
        'feature_importance': importance_df,
        'feature_importance_plot': importance_plot_path,
        'cluster_means': cluster_means,
        'cluster_heatmap': heatmap_path
    }


# 主函数
def main():
    # 设置输出目录
    output_dir = '../outputs/KMeans'
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger(os.path.join(output_dir, 'kmeans_clustering.log'))
    logger.info("===== 学业困难学生识别 - KMeans聚类模型 =====")

    # 加载数据
    logger.info("加载预处理数据...")
    try:
        df = pd.read_csv('../outputs/processed_data.csv')
        logger.info(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return

    # 准备特征和目标
    logger.info("准备特征和目标变量...")
    if 'academic_risk' not in df.columns:
        logger.error("目标变量 'academic_risk' 不存在!")
        return

    X = df.drop(columns=['academic_risk'])
    y = df['academic_risk']
    feature_names = X.columns.tolist()

    # 记录类别分布
    risk_ratio = y.mean()
    logger.info(f"学业困难学生比例: {risk_ratio:.2%} ({y.sum()}/{len(y)})")

    # 划分训练测试集
    logger.info("划分训练集和测试集 (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"训练集: {X_train.shape[0]} 样本")
    logger.info(f"测试集: {X_test.shape[0]} 样本")

    # 处理类别不平衡 - KMeans不直接支持，但我们会通过聚类映射解决

    # 创建管道
    logger.info("创建预处理和建模管道...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(
            n_clusters=2,  # 假设两个聚类：正常学生和学业困难学生
            random_state=42,
            n_init=10,
            max_iter=300
        ))
    ])

    # 训练模型
    logger.info("开始训练KMeans聚类模型...")
    start_train = time.time()
    pipeline.fit(X_train)
    train_time = time.time() - start_train
    logger.info(f"训练完成，耗时: {train_time:.2f}秒")

    # 在训练集上预测聚类标签
    train_clusters = pipeline.named_steps['kmeans'].labels_

    # 创建聚类到类别的映射（基于训练集的真实标签）
    # 找到每个聚类中占多数的类别
    cluster_mapping = {}
    for cluster in np.unique(train_clusters):
        cluster_indices = np.where(train_clusters == cluster)[0]
        majority_class = y_train.iloc[cluster_indices].mode()[0]
        cluster_mapping[cluster] = majority_class

    logger.info(f"聚类到类别的映射: {cluster_mapping}")

    # 创建分类器包装器
    kmeans_classifier = KMeansClassifier(
        kmeans_model=pipeline.named_steps['kmeans'],
        cluster_to_class_mapping=cluster_mapping
    )

    # 创建完整模型（包含预处理和分类器）
    full_model = Pipeline([
        ('scaler', pipeline.named_steps['scaler']),
        ('classifier', kmeans_classifier)
    ])

    # 保存模型
    model_save_path = os.path.join(output_dir, 'kmeans_model.pkl')
    joblib.dump(full_model, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}")

    # 计算轮廓系数
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    silhouette = silhouette_score(X_train_scaled, train_clusters)
    logger.info(f"轮廓系数 (Silhouette Score): {silhouette:.4f}")

    # 评估模型
    logger.info("评估模型性能...")
    metrics = evaluate_model(full_model, X_test, y_test, logger, output_dir)

    # 聚类特征分析
    logger.info("进行聚类特征分析...")
    try:
        cluster_analysis = analyze_clusters(
            model=kmeans_classifier,
            feature_names=feature_names,
            X_train=X_train,
            y_train=y_train,
            logger=logger,
            output_dir=output_dir
        )

        logger.info(f"特征重要性图已保存至: {cluster_analysis['feature_importance_plot']}")
        logger.info(f"聚类热力图已保存至: {cluster_analysis['cluster_heatmap']}")
    except Exception as e:
        logger.error(f"聚类分析失败: {str(e)}")

    # 最终总结
    logger.info("\n===== KMeans聚类模型总结 =====")
    logger.info(f"学业困难学生识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"轮廓系数: {silhouette:.4f}")
    logger.info("学业困难学生识别任务完成!")


if __name__ == "__main__":
    main()