import logging
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 忽略警告
warnings.filterwarnings("ignore")
plt.rc("font", family='Microsoft YaHei', weight="bold")


def setup_logger(log_file):
    """配置日志系统，输出到文件和终端"""
    logger = logging.getLogger('NeuralNetwork')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    output_dir = os.path.dirname(log_file)
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def evaluate_model(model, X_test, y_test, logger, output_dir):
    """评估神经网络模型性能并保存相关结果"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常学生', '学业困难学生'],
                yticklabels=['正常学生', '学业困难学生'])
    plt.title('神经网络 - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    # 分类报告
    report = classification_report(y_test, y_pred, target_names=['正常学生', '学业困难学生'])
    logger.info(f"评估完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info("\n==================== 模型性能指标 ====================")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数 (F1-Score): {f1:.4f}")
    logger.info(f"ROC AUC分数: {roc_auc:.4f}")
    logger.info("\n==================== 混淆矩阵 ====================")
    logger.info(f"\n{cm}")
    logger.info("\n==================== 分类报告 ====================")
    logger.info(f"\n{report}")
    # 保存评估指标
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"评估指标已保存至: {metrics_path}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'confusion_matrix_path': cm_path
    }


def main():
    # 设置输出目录
    output_dir = '../Outputs/NeuralNetwork'
    os.makedirs(output_dir, exist_ok=True)
    # 配置日志
    logger = setup_logger(os.path.join(output_dir, 'neural_network.log'))
    logger.info("===== 学业困难学生识别 - 神经网络模型 =====")
    # 加载数据
    logger.info("加载预处理数据...")
    try:
        df = pd.read_csv('../Outputs/processed_data.csv')
        logger.info(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return
    # 检查目标变量
    if 'academic_risk' not in df.columns:
        logger.error("目标变量 'academic_risk' 不存在!")
        return
    # 准备特征和目标
    drop_cols = ['academic_risk']
    if 'final_result' in df.columns:
        drop_cols.append('final_result')
        logger.info("训练时已去除final_result列，防止信息泄露。")
    X = df.drop(columns=drop_cols)
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
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 构建神经网络模型
    logger.info("训练神经网络模型（MLPClassifier）...")
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # 两层隐藏层
        max_iter=200,
        random_state=42,
        early_stopping=True
    )
    clf.fit(X_train_scaled, y_train)
    # 保存模型
    model_save_path = os.path.join(output_dir, 'neural_network_model.pkl')
    joblib.dump({'scaler': scaler, 'model': clf}, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}")
    # 评估模型
    logger.info("评估模型性能...")
    metrics = evaluate_model(clf, X_test_scaled, y_test, logger, output_dir)
    # 绘制ROC曲线
    logger.info("绘制ROC曲线...")
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('神经网络 - ROC曲线')
    plt.legend()
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC曲线已保存至: {roc_path}")
    # 最终总结
    logger.info("\n===== 神经网络模型总结 =====")
    logger.info(f"学业困难学生识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("学业困难学生识别任务完成!")


if __name__ == "__main__":
    main()
