import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import joblib
import os

font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


# 配置日志系统
def setup_logger(log_file='../outputs/RandomForest/random_forest.log'):
    """配置同时输出到文件和终端的日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 评估函数
def evaluate_model(model, X_test, y_test, logger):
    """全面评估模型性能"""
    start_time = time.time()

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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


# SHAP特征解释分析 - 修复版本
def analyze_with_shap(model, X_train, feature_names, logger, output_dir='../outputs/RandomForest/shap_results'):
    """使用SHAP进行特征解释分析 - 修复形状问题"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 确保X_train是DataFrame格式（SHAP需要特征名）
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train

    # 初始化SHAP解释器 - 使用新的API
    logger.info("初始化SHAP解释器...")
    explainer = shap.Explainer(model, X_train_df)

    # 计算SHAP值（使用部分样本加快计算）
    sample_size = min(1000, X_train_df.shape[0])
    X_sample = X_train_df.sample(sample_size, random_state=42)

    # 计算SHAP值
    logger.info(f"计算 {sample_size} 个样本的SHAP值...")
    shap_values = explainer(X_sample)

    # 检查SHAP值的形状
    logger.info(f"SHAP值形状: {shap_values.values.shape}")
    logger.info(f"数据形状: {X_sample.shape}")

    # 确保形状匹配
    if len(shap_values.values.shape) == 3:
        # 对于二分类问题，取正类（索引1）的SHAP值
        shap_values_pos = shap_values.values[:, :, 1]
        logger.info(f"使用正类SHAP值，新形状: {shap_values_pos.shape}")
    else:
        shap_values_pos = shap_values.values
        logger.info(f"使用完整SHAP值，形状: {shap_values_pos.shape}")

    # 1. SHAP摘要图 - 显示整体特征重要性
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, X_sample, show=False)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()

    # 2. SHAP条形图 - 特征重要性排序
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'shap_bar.png')
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()

    # 3. SHAP依赖图 - 对重要特征的详细分析
    # 计算平均SHAP值以确定最重要的特征
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]  # 选择前3个特征

    dependence_paths = []
    for idx in top_indices:
        feature = feature_names[idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values_pos, X_sample, show=False)
        plt.title(f"{feature} 的SHAP依赖图", fontsize=14)
        dep_path = os.path.join(output_dir, f'shap_dependence_{feature}.png')
        plt.savefig(dep_path, bbox_inches='tight')
        plt.close()
        dependence_paths.append(dep_path)

    # 4. 创建SHAP值数据框用于日志记录
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': np.abs(shap_values_pos).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)

    return {
        'summary_plot': summary_path,
        'bar_plot': bar_path,
        'dependence_plots': dependence_paths,
        'shap_values_df': shap_df
    }


# 主函数
def main():
    # 设置日志
    logger = setup_logger()
    logger.info("===== 学业困难学生识别 - 随机森林算法 (SHAP修复版) =====")

    # 加载数据
    logger.info("加载预处理数据...")
    try:
        df = pd.read_csv('../outputs/processed_data.csv')  # 替换为您的实际文件路径
        logger.info(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return

    # 确保所有分类变量已编码
    logger.info("检查并转换分类变量...")
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
                'age_band', 'disability', 'semester', 'code_module',
                'final_result']

    for col in cat_cols:
        if col in df.columns and df[col].dtype == 'object':
            logger.info(f"对列 {col} 进行标签编码...")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 准备特征和目标
    logger.info("准备特征和目标变量...")

    # 确保删除所有非数值列
    drop_cols = [col for col in df.columns if df[col].dtype == 'object']
    if drop_cols:
        logger.warning(f"删除非数值列: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 检查academic_risk是否存在
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

    # 特征标准化
    logger.info("标准化特征数据...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将标准化后的数据转换回DataFrame以保留特征名
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

    # 初始化随机森林模型
    logger.info("初始化随机森林分类器...")
    rf_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',  # 处理不平衡数据
        'random_state': 42,
        'n_jobs': -1  # 使用所有CPU核心
    }
    model = RandomForestClassifier(**rf_params)

    # 训练模型
    logger.info("开始训练随机森林模型...")
    train_start = time.time()
    model.fit(X_train_scaled_df, y_train)
    train_time = time.time() - train_start
    logger.info(f"训练完成，耗时: {train_time:.2f}秒")

    # 保存模型
    model_save_path = '../outputs/RandomForest/academic_risk_rf_model.pkl'
    joblib.dump(model, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}")

    # 评估模型
    logger.info("评估模型性能...")
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    metrics = evaluate_model(model, X_test_scaled_df, y_test, logger)

    # SHAP特征解释分析
    logger.info("使用SHAP进行特征解释分析...")
    try:
        shap_results = analyze_with_shap(
            model,
            X_train_scaled_df,
            feature_names,
            logger,
            output_dir='../outputs/RandomForest/shap_results'
        )

        logger.info("\n==================== SHAP特征分析结果 ====================")
        logger.info(f"SHAP摘要图已保存至: {shap_results['summary_plot']}")
        logger.info(f"SHAP条形图已保存至: {shap_results['bar_plot']}")

        for plot_path in shap_results['dependence_plots']:
            logger.info(f"SHAP依赖图已保存至: {plot_path}")

        logger.info("\nTop 10 重要特征 (基于SHAP值):")
        logger.info(shap_results['shap_values_df'].head(10).to_string())

        # 解释重要特征的含义
        logger.info("\n重要特征解释:")
        feature_explanations = {
            'overall_active_days': "学生与学习平台互动的总天数，高值表示更活跃的学生",
            'overall_total_clicks': "学生在平台上的总点击次数，反映总体参与度",
            'early_engagement': "课程前4周的平均参与度，是早期预警的关键指标",
            'active_days_homepage': "访问课程主页的天数，反映基本参与程度",
            'active_days_resource': "访问学习资源的天数，反映学习深度",
            'active_days_quiz': "参与测验的天数，反映学习评估参与度",
            'studied_credits': "学生学习的学分数量，可能反映学业负担",
            'num_of_prev_attempts': "之前尝试该课程的次数，重试次数多可能表示学习困难",
            'imd_band': "社会经济地位指数，低值表示社会经济地位较低",
            'highest_education': "最高教育水平，可能影响学习能力",
            'final_result': "最终课程通过情况，Fail或Withdrawn很大程度上表示学业困难",
            'date_unregistration': "课程退课日期，可能与学业困难相关",
        }

        for feature in shap_results['shap_values_df'].head(5)['Feature']:
            if feature in feature_explanations:
                logger.info(f"- {feature}: {feature_explanations[feature]}")
            else:
                logger.info(f"- {feature}: 重要行为特征")
    except Exception as e:
        logger.error(f"SHAP分析失败: {str(e)}")
        logger.info("跳过SHAP分析，继续执行其他任务")

    # 最终总结
    logger.info("\n===== 随机森林模型总结 =====")
    logger.info(f"学业困难学生识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info("学业困难学生识别任务完成!")


if __name__ == "__main__":
    main()