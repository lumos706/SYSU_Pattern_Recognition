import logging
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# 忽略警告
warnings.filterwarnings("ignore")
# 设置全局字体
plt.rc("font", family='Microsoft YaHei', weight="bold")


# 配置日志系统
def setup_logger(log_file):
    """配置同时输出到文件和终端的日志系统"""
    logger = logging.getLogger('SVM')
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


# 评估函数
def evaluate_model(model, X_test, y_test, logger, output_dir):
    """全面评估模型性能"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常学生', '学业困难学生'],
                yticklabels=['正常学生', '学业困难学生'])
    plt.title('SVM - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # ROC曲线
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('SVM - ROC曲线')
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
    else:
        roc_auc = None
        roc_path = None

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=['正常学生', '学业困难学生'])

    # 记录结果
    logger.info(f"评估完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info("\n==================== 模型性能指标 ====================")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数 (F1-Score): {f1:.4f}")
    if roc_auc is not None:
        logger.info(f"ROC AUC分数: {roc_auc:.4f}")

    logger.info("\n==================== 混淆矩阵 ====================")
    logger.info(f"\n{cm}")

    logger.info("\n==================== 分类报告 ====================")
    logger.info(f"\n{report}")

    # 保存评估结果到文件
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
        'roc_curve_path': roc_path,
        'confusion_matrix_path': cm_path
    }


# SHAP特征解释分析
def analyze_with_shap(model, X_train, feature_names, logger, output_dir):
    """使用SHAP进行特征解释分析"""
    shap_dir = os.path.join(output_dir, 'shap_results')
    os.makedirs(shap_dir, exist_ok=True)

    # 确保X_train是DataFrame格式
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        print("X_train转换为DataFrame格式")
    else:
        X_train_df = X_train  # 如果X_train已经是DataFrame，直接使用

    # 采样以加快计算
    X_background = shap.sample(X_train_df, 20, random_state=42)

    # 初始化SHAP解释器
    logger.info("初始化SHAP解释器...")
    explainer = shap.KernelExplainer(model.predict_proba, X_background)

    # 计算SHAP值
    logger.info(f"计算SHAP值...")
    try:
        shap_values = explainer.shap_values(X_background)
        # 检查SHAP值的形状
        logger.info(f"SHAP值形状: {shap_values.shape}")
        logger.info(f"数据集形状: {X_background.shape}")
        # 处理SHAP值形状
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # 取正类的SHAP值
            logger.info(f"使用完整SHAP值形状: {shap_values.shape}")
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # 取正类的SHAP值
            logger.info(f"使用正类SHAP值形状: {shap_values.shape}")
        # 1. SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_background, show=False)
        plt.title("SVM - SHAP特征重要性摘要", fontsize=14)
        plt.tight_layout()
        summary_path = os.path.join(shap_dir, 'shap_summary.png')
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()

        # 2. SHAP条形图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_background, plot_type="bar", show=False)
        plt.title("SVM - SHAP特征重要性排序", fontsize=14)
        plt.tight_layout()
        bar_path = os.path.join(shap_dir, 'shap_bar.png')
        plt.savefig(bar_path, bbox_inches='tight')
        plt.close()

        # 3. 创建SHAP值数据框
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
        }).sort_values('Mean |SHAP|', ascending=False)

        # 保存特征重要性
        shap_csv_path = os.path.join(shap_dir, 'shap_feature_importance.csv')
        shap_df.to_csv(shap_csv_path, index=False)

        # 4. 特征解释
        logger.info("\n==================== SHAP特征分析结果 ====================")
        logger.info(f"Top 10 重要特征 (基于SHAP值):")
        logger.info(shap_df.head(10).to_string())

        # 返回结果
        return {
            'summary_plot': summary_path,
            'bar_plot': bar_path,
            'shap_values_df': shap_df,
            'shap_csv_path': shap_csv_path
        }

    except Exception as e:
        logger.error(f"SHAP计算失败: {str(e)}")
        return None


# 主函数
def main():
    # 设置输出目录
    output_dir = '../outputs/SVM'
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger(os.path.join(output_dir, 'svm_classifier.log'))
    logger.info("===== 学业困难学生识别 - SVM模型 =====")

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

    # 去除final_result，防止信息泄露
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

    # 处理类别不平衡
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # 创建管道
    logger.info("创建预处理和建模管道...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            class_weight=class_weight_dict,
            probability=True,  # 启用概率预测以支持SHAP
            random_state=42
        ))
    ])

    # 参数网格
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['rbf', 'poly']
    }

    # 网格搜索
    logger.info("开始网格搜索优化超参数...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=3,  # 减少cv值以加快计算
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    logger.info("网格搜索完成!")
    logger.info(f"最佳参数: {grid_search.best_params_}")
    logger.info(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 保存模型
    model_save_path = os.path.join(output_dir, 'svm_model.pkl')
    joblib.dump(best_model, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}")

    # 评估模型
    logger.info("评估模型性能...")
    metrics = evaluate_model(best_model, X_test, y_test, logger, output_dir)

    # SHAP分析
    logger.info("使用SHAP进行特征解释分析...")
    try:
        # 获取标准化后的训练数据
        X_train_scaled = best_model.named_steps['scaler'].transform(X_train)

        shap_results = analyze_with_shap(
            best_model.named_steps['classifier'],
            X_train_scaled,
            feature_names,
            logger,
            output_dir
        )

        if shap_results:
            logger.info(f"SHAP摘要图已保存至: {shap_results['summary_plot']}")
            logger.info(f"SHAP条形图已保存至: {shap_results['bar_plot']}")
            logger.info(f"SHAP特征重要性已保存至: {shap_results['shap_csv_path']}")

    except Exception as e:
        logger.error(f"SHAP分析失败: {str(e)}")

    # 最终总结
    logger.info("\n===== SVM模型总结 =====")
    logger.info(f"学业困难学生识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("学业困难学生识别任务完成!")


if __name__ == "__main__":
    main()
