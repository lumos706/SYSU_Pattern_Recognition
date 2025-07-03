import os
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
import shap
from sklearn.tree import export_graphviz
import graphviz
# 忽略警告
warnings.filterwarnings("ignore")
plt.rc("font", family='Microsoft YaHei', weight="bold")


# 配置日志系统
def setup_logger(log_file):
    """配置同时输出到文件和终端的日志系统"""
    logger = logging.getLogger('DecisionTree')
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


# 评估函数
def evaluate_model(model, X_test, y_test, logger, output_dir):
    """全面评估模型性能"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常学生', '学业困难学生'],
                yticklabels=['正常学生', '学业困难学生'])
    plt.title('决策树 - 混淆矩阵')
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
        'confusion_matrix_path': cm_path
    }


# 特征重要性分析
def analyze_with_shap(model, X_train, feature_names, logger, output_dir):
    """使用SHAP进行特征解释分析"""
    shap_dir = os.path.join(output_dir, 'shap_results')
    os.makedirs(shap_dir, exist_ok=True)
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train

    logger.info("初始化SHAP解释器...")
    explainer = shap.Explainer(model.named_steps['classifier'], X_train_df)

    sample_size = min(1000, X_train_df.shape[0])
    X_sample = X_train_df.sample(sample_size, random_state=42)

    logger.info(f"计算 {sample_size} 个样本的SHAP值...")
    shap_values = explainer(X_sample)

    # 处理SHAP值形状
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values.values[:, :, 1]  # 取正类的SHAP值
    else:
        shap_values = shap_values.values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    summary_path = os.path.join(shap_dir, 'shap_summary.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    bar_path = os.path.join(shap_dir, 'shap_bar.png')
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)

    shap_csv_path = os.path.join(shap_dir, 'shap_feature_importance.csv')
    shap_df.to_csv(shap_csv_path, index=False)

    logger.info(f"SHAP摘要图已保存至: {summary_path}")
    logger.info(f"SHAP条形图已保存至: {bar_path}")
    logger.info("\nTop 10 重要特征 (基于SHAP值):")
    logger.info(shap_df.head(10).to_string())

    return shap_df


# 决策树可视化
def visualize_tree(model, feature_names, output_dir):
    estimator = model.named_steps['classifier']
    dot_path = os.path.join(output_dir, 'decision_tree.dot')
    export_graphviz(
        estimator,
        out_file=dot_path,
        feature_names=feature_names,
        class_names=['正常学生', '学业困难学生'],
        filled=True, rounded=True,
        special_characters=True,
        fontname="Microsoft YaHei"
    )
    with open(dot_path, encoding='utf-8') as f:
        dot_graph = f.read()
    # 替换字体，确保 dot 文件里所有字体都为中文字体
    dot_graph = dot_graph.replace('helvetica', 'Microsoft YaHei')
    graph = graphviz.Source(dot_graph)
    svg_path = os.path.join(output_dir, 'decision_tree_graphviz')
    graph.render(svg_path, format='svg', cleanup=True)
    return svg_path


# 主函数
def main():
    # 设置输出目录
    output_dir = '../Outputs/DecisionTree'
    os.makedirs(output_dir, exist_ok=True)
    # 设置日志
    logger = setup_logger(os.path.join(output_dir, 'decision_tree.log'))
    logger.info("===== 学业困难学生识别 - 决策树模型 =====")
    # 加载数据
    logger.info("加载预处理数据...")
    try:
        df = pd.read_csv('../Outputs/processed_data.csv')
        logger.info(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return
    # 准备特征和目标
    logger.info("准备特征和目标变量...")
    if 'academic_risk' not in df.columns:
        logger.error("目标变量 'academic_risk' 不存在!")
        return
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
    # 创建管道
    logger.info("创建预处理和建模管道...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])
    # 参数网格
    param_grid = {
        'classifier__max_depth': [3, 5, 7, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    # 网格搜索
    logger.info("开始网格搜索优化超参数...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logger.info("网格搜索完成!")
    logger.info(f"最佳参数: {grid_search.best_params_}")
    logger.info(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    # 保存模型
    model_save_path = os.path.join(output_dir, 'decision_tree_model.pkl')
    joblib.dump(best_model, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}")
    # 评估模型
    logger.info("评估模型性能...")
    metrics = evaluate_model(best_model, X_test, y_test, logger, output_dir)
    # 特征重要性分析
    logger.info("使用SHAP进行特征解释分析...")
    shap_df = analyze_with_shap(best_model, X_train, feature_names, logger, output_dir)
    # 决策树结构可视化
    logger.info("保存决策树结构图...")
    tree_path = visualize_tree(best_model, feature_names, output_dir)
    logger.info(f"决策树结构图已保存至: {tree_path}")
    # ROC曲线
    if metrics['roc_auc'] is not None:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('决策树 - ROC曲线')
        plt.legend()
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC曲线已保存至: {roc_path}")
    # 最终总结
    logger.info("\n===== 决策树模型总结 =====")
    logger.info(f"学业困难学生识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("学业困难学生识别任务完成!")


if __name__ == "__main__":
    main()
