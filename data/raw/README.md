# README_Data.md

## 1. Dataset Name

Breast Cancer Wisconsin Diagnostic

## 2. Dataset Source

Dataset được lấy từ UCI Machine Learning Repository.

Official link:  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

DOI:  
https://doi.org/10.24432/C5DW2B

## 3. Reason for Choosing This Dataset

Dataset này được chọn vì phù hợp với yêu cầu của task:

- Là bài toán binary classification.
- Dataset nhỏ, dễ chạy demo nhanh.
- Có mất cân bằng class ở mức vừa phải.
- Feature đều là numerical values, không cần xử lý categorical phức tạp.
- Nguồn dữ liệu rõ ràng, public, dễ reproduce.
- Phù hợp để demo ADASYN oversampling.

## 4. Dataset Description

Breast Cancer Wisconsin Diagnostic là dataset dùng để phân loại khối u là malignant hoặc benign dựa trên các đặc trưng được tính từ ảnh tế bào.

Thông tin chính:

- Number of samples: 569
- Number of features: 30
- Task type: Binary classification
- Missing values: No
- Target column: Diagnosis

## 5. Target Column and Class Labels

### Raw UCI target column

Target column: `Diagnosis`

Class labels:

| Label | Meaning |
|---|---|
| M | Malignant |
| B | Benign |

### Demo mapping

Trong demo Python/scikit-learn:

| Value | Meaning |
|---|---|
| 0 | Malignant |
| 1 | Benign |

## 6. Original Class Distribution

| Class | Count | Percentage | Class Type |
|---|---:|---:|---|
| Benign | 357 | 62.74% | Majority class |
| Malignant | 212 | 37.26% | Minority class |

Majority class: Benign  
Minority class: Malignant  
Majority/minority ratio: approximately 1.68:1

## 7. How to Place Files

Recommended folder structure:

```text
project_root/
├── README_Data.md
├── data/
│   ├── raw/
│   │   └── uci_wdbc/
│   │       ├── wdbc.data
│   │       └── wdbc.names
│   ├── interim/
│   │   └── wdbc_with_target.csv
│   └── sample/
│       └── sample_wdbc_8rows.csv
├── notebooks/
│   └── adasyn_demo.ipynb
└── outputs/
    ├── class_distribution_before_after.csv
    └── model_metrics_comparison.csv