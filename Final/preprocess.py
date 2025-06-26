import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 配置日志
# 配置日志
LOG_FILE = './outputs/preprocess.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),  # 输出到文件
        logging.StreamHandler()  # 可选：同时输出到终端
    ]
)
_LOG = logging.getLogger(__name__)


def load_and_merge_data(base_path='oulad'):
    """加载并合并所有数据集"""
    tables = {
        'assessments': 'assessments.csv',
        'courses': 'courses.csv',
        'studentAssessment': 'studentAssessment.csv',
        'studentInfo': 'studentInfo.csv',
        'studentRegistration': 'studentRegistration.csv',
        'studentVLE': 'studentVLE.csv',
        'vle': 'vle.csv'
    }

    data_dict = {}
    for name, file in tables.items():
        try:
            data_dict[name] = pd.read_csv(os.path.join(base_path, file))
            _LOG.info(f"Loaded {name}: {data_dict[name].shape}")
        except Exception as e:
            _LOG.error(f"Error loading {file}: {str(e)}")
            raise

    # 合并学生信息
    merged = data_dict['studentInfo'].merge(
        data_dict['studentRegistration'],
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )

    # 合并课程信息
    merged = merged.merge(
        data_dict['courses'],
        on=['code_module', 'code_presentation'],
        how='left'
    )

    return merged, data_dict


def handle_missing_data(df):
    """处理缺失值"""
    _LOG.info("Handling missing data...")

    # 记录初始缺失情况
    initial_nulls = df.isnull().sum()

    # 删除全空列
    drop_cols = [col for col in df.columns if df[col].isnull().all()]
    if drop_cols:
        _LOG.warning(f"Dropping completely null columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 分类列：删除缺失行
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    df = df.dropna(subset=cat_cols)

    # 数值列：填充0
    num_cols = ['num_of_prev_attempts', 'studied_credits', 'date_registration', 'date_unregistration']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 记录处理结果
    final_nulls = df.isnull().sum().sum()
    _LOG.info(f"Missing values reduced from {initial_nulls.sum()} to {final_nulls}")
    return df


def process_vle_data(df, vle_data, student_vle_data):
    """处理VLE点击数据"""
    _LOG.info("Processing VLE data...")

    # 合并VLE信息
    vle_merged = student_vle_data.merge(
        vle_data[['id_site', 'activity_type']],
        on='id_site',
        how='left'
    ).dropna(subset=['activity_type'])

    # 聚合VLE数据
    vle_agg = vle_merged.groupby(
        ['code_module', 'code_presentation', 'id_student', 'activity_type']
    ).agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique')
    ).reset_index()

    # 创建透视表
    vle_pivot = vle_agg.pivot_table(
        index=['code_module', 'code_presentation', 'id_student'],
        columns='activity_type',
        values=['total_clicks', 'active_days'],
        fill_value=0
    )

    # 扁平化列名
    vle_pivot.columns = [f'{col[0]}_{col[1]}' for col in vle_pivot.columns]
    vle_pivot = vle_pivot.reset_index()

    # 合并到主数据
    df = df.merge(
        vle_pivot,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )

    # 填充缺失的VLE数据
    vle_cols = [col for col in df.columns if 'total_clicks_' in col or 'active_days_' in col]
    df[vle_cols] = df[vle_cols].fillna(0)

    return df


def handle_outliers(df, output_dir):
    """处理异常值并可视化"""
    _LOG.info("Handling outliers...")

    # 识别数值列
    num_cols = ['num_of_prev_attempts', 'studied_credits', 'date_registration',
                'date_unregistration', 'module_presentation_length']
    vle_cols = [col for col in df.columns if 'total_clicks_' in col or 'active_days_' in col]
    num_cols.extend(vle_cols)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 绘制原始异常值箱线图
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df[num_cols])
    plt.xticks(rotation=90)
    plt.title('Outliers Before Capping')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outliers_before.png'))
    plt.close()

    # 应用98%分位截断
    for col in num_cols:
        upper_limit = df[col].quantile(0.98)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    # 绘制处理后的箱线图
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df[num_cols])
    plt.xticks(rotation=90)
    plt.title('Outliers After Capping')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outliers_after.png'))
    plt.close()

    return df


def feature_engineering(df):
    """特征工程"""
    _LOG.info("Performing feature engineering...")

    # 从课程代码中提取年份和学期
    df['presentation_year'] = '20' + df['code_presentation'].str[:2]
    df['semester'] = df['code_presentation'].str[2].map({'B': 'Spring', 'J': 'Autumn'})

    # 创建总体互动指标
    click_cols = [col for col in df.columns if 'total_clicks_' in col]
    active_cols = [col for col in df.columns if 'active_days_' in col]

    df['overall_total_clicks'] = df[click_cols].sum(axis=1)
    df['overall_active_days'] = df[active_cols].sum(axis=1)
    df['avg_daily_clicks'] = np.where(
        df['overall_active_days'] > 0,
        df['overall_total_clicks'] / df['overall_active_days'],
        0
    )

    # 创建早期参与特征 (前4周)
    early_cols = [
        'active_days_homepage', 'total_clicks_homepage',
        'active_days_quiz', 'total_clicks_quiz',
        'active_days_resource', 'total_clicks_resource'
    ]

    df['early_engagement'] = df[early_cols].mean(axis=1)

    return df


def encode_categorical(df):
    """编码分类变量 - 修正版本"""
    _LOG.info("Encoding categorical variables...")

    # 定义学业困难标准 (Fail或Withdrawn)
    df['academic_risk'] = df['final_result'].apply(
        lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0
    )

    # 需要编码的分类列
    cat_cols = ['gender', 'code_presentation', 'region', 'highest_education', 'imd_band',
                'age_band', 'disability', 'semester', 'code_module',
                'final_result']  # 添加了final_result

    le_dict = {}
    for col in cat_cols:
        # 检查列是否在DataFrame中
        if col in df.columns:
            # 确保值为字符串类型
            df[col] = df[col].astype(str)
            le = preprocessing.LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

            # 输出类别映射到日志
            mapping = {category: index for index, category in enumerate(le.classes_)}
            _LOG.info(f"Column '{col}' encoding mapping: {mapping}")
        else:
            _LOG.warning(f"列 {col} 不存在于DataFrame中，跳过编码")

    return df, le_dict


def preprocess_pipeline(output_base='outputs'):
    """完整预处理流程"""
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    plot_dir = os.path.join(output_base, 'preprocess_plots', 'cleaning')

    # 1. 加载数据
    merged_df, data_dict = load_and_merge_data()

    # 2. 处理缺失值
    cleaned_df = handle_missing_data(merged_df)

    # 3. 添加VLE数据
    with_vle_df = process_vle_data(cleaned_df, data_dict['vle'], data_dict['studentVLE'])

    # 4. 处理异常值
    outlier_handled_df = handle_outliers(with_vle_df, plot_dir)

    # 5. 特征工程
    feature_df = feature_engineering(outlier_handled_df)

    # 6. 编码分类变量
    final_df, encoders = encode_categorical(feature_df)

    # 保存处理后的数据
    final_df.to_csv(os.path.join(output_base, 'processed_data.csv'), index=False)
    _LOG.info(f"Final dataset shape: {final_df.shape}")

    return final_df, encoders


def prepare_model_data(df, test_size=0.2, balance_data=True):
    """准备建模数据"""
    # 分离特征和目标
    X = df.drop(columns=['academic_risk', 'final_result', 'id_student', 'code_presentation'])
    y = df['academic_risk']

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # 处理不平衡数据
    if balance_data:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        _LOG.info(f"After SMOTE - Train shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")

    # 标准化特征
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }


if __name__ == "__main__":
    # 运行预处理流程
    processed_df, encoders = preprocess_pipeline()

    # 准备建模数据
    model_data = prepare_model_data(processed_df)

    # 打印结果摘要
    _LOG.info(f"Training set size: {model_data['X_train'].shape}")
    _LOG.info(f"Testing set size: {model_data['X_test'].shape}")
    _LOG.info(f"Risk ratio in train: {model_data['y_train'].mean():.2f}")
    _LOG.info(f"Risk ratio in test: {model_data['y_test'].mean():.2f}")