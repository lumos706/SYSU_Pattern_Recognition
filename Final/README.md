# Final 模式识别项目说明

## 目录结构

```
Final/
├── preprocess.py                # 数据预处理脚本
├── Model/                       # 各类模型代码
│   ├── Compare.py               # 模型性能对比与可视化
│   ├── KmeansClustering.py      # K-Means 聚类模型
│   ├── LogisticRegression.py    # 逻辑回归模型
│   ├── RandomForest.py          # 随机森林模型
│   └── SVM.py                   # 支持向量机模型
├── oulad/                       # 原始数据集（csv）
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
└── Outputs/                     # 输出结果
    ├── processed_data.csv       # 预处理后数据
    ├── Comparison/              # 模型对比结果
    │   ├── model_comparison_report.txt
    │   ├── model_comparison.csv
    │   ├── model_comparison.log
    │   └── Comparison_plots/
    │       ├── metrics_comparison.png
    │       └── overall_comparison.png
    ├── KMeans/                  # KMeans 相关输出
    │   ├── confusion_matrix.png
    │   ├── kmeans_clustering.log
    │   ├── kmeans_model.pkl
    │   ├── metrics.csv
    │   └── cluster_analysis/
    ├── LogisticRegression/      # 逻辑回归相关输出
    │   ├── confusion_matrix.png
    │   ├── logistic_regression_model.pkl
    │   ├── logistic_regression.log
    │   ├── metrics.csv
    │   └── shap_results/
    ├── RandomForest/            # 随机森林相关输出
    │   ├── academic_risk_rf_model.pkl
    │   ├── metrics.csv
    │   ├── random_forest.log
    │   └── shap_results/
    ├── SVM/                     # SVM 相关输出
    │   ├── confusion_matrix.png
    │   ├── metrics.csv
    │   ├── roc_curve.png
    │   ├── svm_classifier.log
    │   ├── svm_model.pkl
    │   └── shap_results/
    └── Preprocess/              # 预处理日志等
        ├── preprocess.log
        └── cleaning/
```

## 使用流程

1. **数据预处理**  
   运行 `preprocess.py`，对原始数据进行清洗和特征工程，生成 `Outputs/processed_data.csv`。

2. **模型训练与评估**  
   依次运行 `Model` 目录下的各模型脚本（如 `KmeansClustering.py`、`LogisticRegression.py` 等），每个脚本会读取预处理数据，训练模型，并将评估指标、模型文件、可视化图等输出到 `Outputs/模型名/` 文件夹。

3. **模型性能对比**  
   运行 `Model/Compare.py`，自动收集所有模型的评估指标，生成对比报告和可视化图表，输出到 `Outputs/Comparison/`。

## 扩展规范

- **新模型添加**  
  1. 在 `Model/` 目录下新建 `YourModelName.py`，实现训练、评估、输出指标（如 `metrics.csv`）、模型文件（如 `your_model.pkl`）和可视化图。
  2. 在 `Outputs/` 下为新模型建立同名文件夹，存放输出内容。
  3. 确保 `metrics.csv` 格式与现有模型一致（包含 `Metric`, `Value` 两列，便于对比）。
  4. 如需参与自动对比，在 `Compare.py` 的 `models` 字典中添加新模型及其中文名。

- **输出规范**  
  - 每个模型输出应包括：`metrics.csv`（主要评估指标）、模型文件（如 `.pkl`）、主要可视化图（如混淆矩阵、ROC 曲线等）、日志文件。
  - 所有输出均存放于 `Outputs/模型名/` 下，便于统一管理和后续对比。

## 依赖环境

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn 等

可通过如下命令安装依赖：

```
pip install -r requirements.txt
```

（请根据实际情况补充 `requirements.txt`）

## 贡献说明

- 保持代码风格一致，注释清晰。
- 输出文件命名规范，便于自动化收集和对比。
- 如有改动，及时更新本说明文档。

---

如有疑问请联系项目负责人或在小组群内讨论。