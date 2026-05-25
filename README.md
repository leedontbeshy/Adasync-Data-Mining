# ADASYN Demo

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
│   └── quy_dinh_pipeline_thuc_nghiem.md
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py
└── results/
```

## 2. Các thư mục và file chính

### `data/`

Chứa dữ liệu dùng cho notebook.

- `data/raw/`: dữ liệu gốc, giữ nguyên theo nguồn tải về.
- `data/raw/README.md`: mô tả nguồn dataset, lý do chọn dataset, target column và class distribution ban đầu.
- `data/raw/uci_wdbc/wdbc.data`: file dữ liệu chính được notebook đọc.
- `data/raw/uci_wdbc/wdbc.names`: mô tả gốc của dataset từ UCI.

### `src/`

Chứa các hàm hỗ trợ đúng phạm vi pipeline của Phúc.

- `src/__init__.py`: giúp Python nhận `src` là package để notebook import được.
- `src/data_preprocessing.py`: chứa hàm tách `X/y`, train/test split, preprocessing bằng imputer + scaler, và checklist chống data leakage.

### `docs/`

Chứa tài liệu viết để đưa vào báo cáo hoặc tiểu luận.

- `docs/quy_dinh_pipeline_thuc_nghiem.md`: quy định train/test split, `stratify`, `random_state`, preprocessing, model và metric đánh giá.

### `results/`

Chứa kết quả sau khi các phần thí nghiệm được hoàn thiện.

Hiện tại thư mục này chỉ có `.gitkeep` để giữ thư mục rỗng. Sau khi nhóm chạy đủ các phương pháp, các file như bảng metric, confusion matrix và biểu đồ so sánh sẽ được lưu ở đây.

### File ở thư mục gốc

- `demo_adasyn.ipynb`: notebook khung theo phân công từng thành viên.
- `requirements.txt`: danh sách thư viện cần cài để chạy notebook.
- `README.md`: hướng dẫn cấu trúc project, cách cài thư viện, cách chạy notebook và quy tắc pipeline.

Dataset được đọc từ:

```text
data/raw/uci_wdbc/wdbc.data
```

Trong file dữ liệu gốc:

- Cột 1: `id`, không đưa vào quá trình huấn luyện.
- Cột 2: `diagnosis`, gồm hai nhãn `B` và `M`.
- 30 cột còn lại: các đặc trưng dạng số.
- Nhãn minority/positive dùng để đánh giá: `M` (Malignant).

## 3. Cài thư viện

Chạy lệnh sau từ thư mục gốc của project:

```powershell
python -m pip install -r requirements.txt
```

## 4. Chạy notebook

Từ thư mục gốc của project:

```powershell
jupyter notebook demo_adasyn.ipynb
```

Nếu lệnh `jupyter` chưa có trong `PATH`, dùng:

```powershell
python -m jupyter notebook demo_adasyn.ipynb
```

Notebook hiện có các phần chính theo người phụ trách:

- Phúc: import, cấu hình pipeline, preprocessing, train/test split và checklist chống data leakage.
- Quang: load data, kiểm tra dữ liệu đầu vào, EDA và class distribution.
- Quân: Baseline, Random Oversampling, Random Undersampling.
- Phước: SMOTE, ADASYN và thử `k_neighbors`.
- Quyên: metrics, confusion matrix và biểu đồ so sánh.

## 5. Quy tắc pipeline đã chốt

Chi tiết đầy đủ nằm trong `docs/quy_dinh_pipeline_thuc_nghiem.md`.

- Split train/test trước mọi bước resampling.
- Dùng `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Không đưa cột `id` vào training.
- Ép toàn bộ feature về dạng số.
- Nếu có missing values, impute bằng median.
- Scale feature bằng `StandardScaler`.
- Fit preprocessing chỉ trên train; test set chỉ được transform.
- Test set không được oversample, undersample, SMOTE hoặc ADASYN.
- Tất cả metric cuối cùng phải đánh giá trên test set gốc.

## 6. Output kỳ vọng

Sau khi các thành viên phụ trách điền tiếp phần của mình, thư mục `results/` có thể chứa:

```text
metrics_comparison.csv
confusion_matrix_baseline.png
confusion_matrix_smote.png
confusion_matrix_adasyn.png
metric_comparison_chart.png
```
