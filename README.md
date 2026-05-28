# ADASYN Demo

Project này là notebook thực nghiệm cho bài toán phân loại Breast Cancer Wisconsin Diagnostic. Mục tiêu là so sánh Baseline, Random Oversampling, Random Undersampling, SMOTE và ADASYN trên cùng một pipeline để đánh giá ảnh hưởng của các kỹ thuật xử lý mất cân bằng dữ liệu.

Quy tắc quan trọng nhất: split train/test trước, preprocessing chỉ fit trên train, resampling chỉ chạy trên train, và test set gốc chỉ dùng để đánh giá cuối cùng.

## 1. Cấu trúc thư mục

```text
project_root/
├── demo_adasyn.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── README.md
│       └── uci_wdbc/
│           ├── wdbc.data
│           └── wdbc.names
├── docs/
│   ├── evaluation_metrics_and_result_table_schema.md
│   ├── preprocessing_train_test.md
│   └── quy_dinh_pipeline_thuc_nghiem.md
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py
└── results/
    ├── metrics_comparison.csv
    ├── confusion_matrix_*.png
    ├── metrics_comparison_*.png
    ├── roc_curves_comparison.png
    ├── pr_curves_comparison.png
    └── train_test_class_distribution.csv
```

## 2. Dataset

Notebook đọc dữ liệu từ:

```text
data/raw/uci_wdbc/wdbc.data
```

Nếu clone project mới, đặt file UCI WDBC vào đúng thư mục:

```text
data/raw/uci_wdbc/
├── wdbc.data
└── wdbc.names
```

Thông tin chính:

- Dataset: Breast Cancer Wisconsin Diagnostic.
- Target column: `diagnosis`.
- Class labels: `B` = Benign, `M` = Malignant.
- Positive/minority class dùng để tính metric: `M`.
- Feature dùng để train: 30 cột số, không dùng cột `id`.
- Phân phối ban đầu: `B` = 357 mẫu, `M` = 212 mẫu.

## 3. Cài thư viện

Khuyến nghị tạo môi trường ảo trước khi cài thư viện. Chạy các lệnh sau từ thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu không dùng virtual environment, vẫn có thể cài trực tiếp:

```powershell
python -m pip install -r requirements.txt
```

## 4. Chạy notebook

Mở notebook bằng Jupyter:

```powershell
python -m jupyter notebook demo_adasyn.ipynb
```

Trong Jupyter, chọn `Kernel -> Restart & Run All` để chạy lại toàn bộ từ đầu. Notebook đã dùng đường dẫn tương đối theo thư mục project, nên nên mở Jupyter từ thư mục gốc này.

Có thể chạy notebook dạng headless để kiểm tra nhanh:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace demo_adasyn.ipynb
```

## 5. Pipeline thực nghiệm

Pipeline chuẩn nằm trong `src/data_preprocessing.py` và được mô tả thêm ở `docs/quy_dinh_pipeline_thuc_nghiem.md`.

- Tách `X/y`, loại `id` khỏi feature.
- Chia dữ liệu bằng `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Fit `SimpleImputer(strategy="median")` và `StandardScaler` trên train.
- Transform test bằng preprocessor đã fit trên train.
- Train Logistic Regression với cùng cấu hình cho mọi phương pháp.
- Chỉ resampling trên `X_train_processed, y_train`.
- Đánh giá tất cả phương pháp trên `X_test_processed, y_test`.

## 6. Kiểm tra data leakage

Notebook có phần `Phúc - Checklist chống data leakage` để kiểm tra:

- Train/test indices tách biệt.
- Preprocessing không làm đổi số dòng train/test.
- Test set chỉ được transform, không bị resampling.
- `fit_resample` xuất hiện sau cell split.
- Không có lệnh `fit_resample` nào dùng `X_test` hoặc `y_test`.
- Test features và test labels không thay đổi sau các cell resampling.

Khi thêm phương pháp mới, dùng mẫu sau:

```python
sampler = SomeSampler(random_state=RANDOM_STATE)
X_train_new, y_train_new = sampler.fit_resample(X_train_processed, y_train)
model.fit(X_train_new, y_train_new)
y_pred = model.predict(X_test_processed)
```

Không dùng `fit_resample` với `X_test_processed` hoặc `y_test`.

## 7. Đọc kết quả

Bảng metric chính nằm ở:

```text
results/metrics_comparison.csv
```

Các cột trong bảng:

- `Method`: tên phương pháp.
- `Precision`: precision của lớp `M`.
- `Recall`: recall của lớp `M`.
- `F1-score`: F1-score của lớp `M`.
- `ROC-AUC`: khả năng phân tách hai lớp trên nhiều ngưỡng.
- `PR-AUC`: Average Precision, hữu ích cho dữ liệu mất cân bằng.

Các hình kết quả chính:

- `results/confusion_matrix_baseline.png`
- `results/confusion_matrix_random_oversampling.png`
- `results/confusion_matrix_random_undersampling.png`
- `results/confusion_matrix_smote.png`
- `results/confusion_matrix_adasyn.png`
- `results/metrics_comparison_bars.png`
- `results/metrics_comparison_grouped.png`
- `results/metrics_comparison_heatmap.png`
- `results/roc_curves_comparison.png`
- `results/pr_curves_comparison.png`

Nhận xét nhanh theo kết quả hiện tại: các phương pháp resampling cải thiện Recall/F1 của lớp `M` so với Baseline, nhưng ADASYN không vượt SMOTE về ROC-AUC và PR-AUC trên split này. Vì vậy kết luận nên nhấn mạnh trade-off giữa Precision, Recall, PR-AUC và đặc điểm dữ liệu, không kết luận ADASYN luôn tốt nhất.

