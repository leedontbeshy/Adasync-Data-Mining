# Preprocessing và train/test split

## 1. Mục tiêu

chuẩn hóa dữ liệu đầu vào trước khi các thành viên khác chạy Baseline, Random Oversampling, Random Undersampling, SMOTE và ADASYN.

Các yêu cầu chính:

- Xử lý missing values nếu có.
- Encode categorical features nếu dataset có categorical features.
- Scale numeric features vì mô hình Logistic Regression và các kỹ thuật dựa trên khoảng cách như SMOTE/ADASYN nhạy với thang đo.
- Chia train/test theo tỷ lệ 80/20 bằng stratify.
- Cố định `random_state=42` để kết quả có thể tái lập.
- Lưu lại tỷ lệ class của full/train/test để kiểm tra việc stratify.
- Đảm bảo không data leakage giữa train set và test set.

## 2. Input

Task này chạy sau phần Load data và EDA của Quang trong `demo_adasyn.ipynb`.

Các biến đầu vào cần có:

| Biến | Ý nghĩa |
|---|---|
| `df` | DataFrame dữ liệu WDBC đã load từ `data/raw/uci_wdbc/wdbc.data` |
| `FEATURE_COLUMNS` | Danh sách feature, đã loại `id` và `diagnosis` |
| `TARGET_COLUMN` | Tên cột target, hiện là `diagnosis` |

Trong dataset WDBC:

- Cột `id` chỉ là mã định danh, không đưa vào training.
- Cột target là `diagnosis`, gồm hai nhãn `B` và `M`.
- Có 30 feature, tất cả đều là numeric.
- Không có missing values trong dữ liệu gốc, nhưng pipeline vẫn giữ `SimpleImputer` để đúng quy định thực nghiệm.

## 3. Quy trình thực hiện

### 3.1. Tách feature và target

Notebook gọi hàm:

```python
X, y = make_features_target(df, FEATURE_COLUMNS, TARGET_COLUMN)
```

Kết quả:

- `X`: chỉ chứa các feature dùng cho training.
- `y`: chứa nhãn `diagnosis`.
- `id` không xuất hiện trong `X`.

### 3.2. Kiểm tra kiểu feature và missing values

Notebook kiểm tra:

- Số lượng numeric feature.
- Số lượng categorical feature.
- Tổng missing values trước khi split.
- Số cột có missing values.
- Việc loại `id` khỏi training.

Với WDBC hiện tại:

| Nội dung | Giá trị |
|---|---:|
| Feature columns used for training | 30 |
| Numeric feature columns | 30 |
| Categorical feature columns | 0 |
| Total missing values before split | 0 |
| Columns with missing values | 0 |

Vì không có categorical feature nên không cần encode trong dataset này. Tuy nhiên helper trong `src/data_preprocessing.py` đã hỗ trợ categorical feature nếu sau này dữ liệu thay đổi.

### 3.3. Chia train/test

Notebook gọi:

```python
X_train, X_test, y_train, y_test = split_data(X, y)
```

Hàm `split_data` dùng:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
```

Ý nghĩa:

- `test_size=0.2`: test set chiếm 20%, train set chiếm 80%.
- `stratify=y`: giữ tỷ lệ class của train/test gần giống dữ liệu gốc.
- `random_state=42`: cố định kết quả chia dữ liệu.

Kích thước sau split:

| Tập dữ liệu | Shape |
|---|---:|
| `X_train` | `(455, 30)` |
| `X_test` | `(114, 30)` |

### 3.4. Preprocessing train/test

Notebook gọi:

```python
X_train_processed, X_test_processed, preprocessor = preprocess_train_test(
    X_train,
    X_test,
)
```

Pipeline preprocessing nằm trong `src/data_preprocessing.py`.

Với numeric features:

```python
Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
```

Với categorical features nếu có:

```python
Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
```

Quy tắc chống data leakage:

- `preprocessor.fit_transform(X_train)` chỉ fit trên train set.
- `preprocessor.transform(X_test)` chỉ transform test set bằng bộ biến đổi đã fit trên train.
- Không fit imputer/scaler/encoder trên toàn bộ dataset.
- Không resampling trên test set.

Với WDBC hiện tại, shape sau preprocessing:

| Tập dữ liệu | Shape |
|---|---:|
| `X_train_processed` | `(455, 30)` |
| `X_test_processed` | `(114, 30)` |

## 4. Tỷ lệ class sau stratified split

Notebook lưu tỷ lệ class vào:

```text
results/train_test_class_distribution.csv
```

Nội dung hiện tại:

| Dataset | Class | Label | Count | Percentage |
|---|---|---|---:|---:|
| full | B | Benign | 357 | 62.74 |
| full | M | Malignant | 212 | 37.26 |
| train | B | Benign | 285 | 62.64 |
| train | M | Malignant | 170 | 37.36 |
| test | B | Benign | 72 | 63.16 |
| test | M | Malignant | 42 | 36.84 |

Nhận xét:

- Tỷ lệ class của train/test gần giống tỷ lệ class ban đầu.
- Lớp `M` vẫn là minority/positive class.
- Split đạt yêu cầu cho bài toán mất cân bằng lớp.

## 5. Output cho các task sau

Sau khi chạy cell của Phúc, notebook tạo các biến để những phần sau dùng tiếp:

| Biến | Ý nghĩa |
|---|---|
| `X_train` | Train features gốc, chưa scale |
| `X_test` | Test features gốc, chưa scale |
| `y_train` | Train labels |
| `y_test` | Test labels |
| `X_train_processed` | Train features sau impute/encode/scale |
| `X_test_processed` | Test features sau impute/encode/scale |
| `preprocessor` | Pipeline preprocessing đã fit trên train set |
| `class_distribution_train_test` | Bảng tỷ lệ class full/train/test |
| `train_test_class_distribution_path` | Đường dẫn file CSV lưu tỷ lệ class |

Các phần Baseline, Random Oversampling, Random Undersampling, SMOTE và ADASYN nên dùng:

- `X_train_processed`, `y_train` để huấn luyện hoặc resampling trên train.
- `X_test_processed`, `y_test` để đánh giá cuối cùng.

Không được resampling trên `X_test_processed` hoặc `y_test`.

## 6. Kiểm tra đã thực hiện

Notebook có các assert cơ bản:

- Train/test index không bị trùng.
- Train sau preprocessing không còn missing values.
- Test sau preprocessing không còn missing values.
- Cột sau preprocessing của train và test khớp nhau.

Ngoài ra, cell checklist chống data leakage kiểm tra:

- Train/test indices tách biệt.
- Preprocessing giữ nguyên số dòng train.
- Preprocessing giữ nguyên số dòng test.
- Feature columns sau preprocessing của train/test khớp nhau.
- Test set chỉ được transform, không bị resampling.

## 7. Kết luận

Phần preprocessing và train/test split đã đúng quy định pipeline thực nghiệm:

- Đã loại `id` khỏi training.
- Đã kiểm tra missing values và categorical features.
- Đã chuẩn hóa numeric features bằng `StandardScaler`.
- Đã dùng median imputer cho numeric features.
- Đã split train/test theo `80/20`, `stratify=y`, `random_state=42`.
- Đã lưu tỷ lệ class của full/train/test.
- Đã fit preprocessing chỉ trên train set để tránh data leakage.
