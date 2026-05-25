# Quy định pipeline thực nghiệm

Trong quá trình thực nghiệm, toàn bộ phương pháp được so sánh trên cùng một pipeline để đảm bảo kết quả công bằng và có thể tái lập. Các bước tiền xử lý, chia dữ liệu, huấn luyện và đánh giá được quy định thống nhất trước khi chạy Baseline, Random Oversampling, Random Undersampling, SMOTE và ADASYN.

## 1. Train/test split

Dữ liệu được chia thành hai tập:

- Train set: 80%
- Test set: 20%

Việc chia dữ liệu được thực hiện bằng `train_test_split` với tham số:

```python
test_size = 0.2
stratify = y
random_state = 42
```

Tham số `stratify=y` được sử dụng để giữ tỷ lệ giữa các lớp trong train set và test set gần giống với tỷ lệ lớp ban đầu của toàn bộ dataset. Đây là yêu cầu quan trọng trong bài toán mất cân bằng lớp, vì nếu không stratify, tập test có thể không phản ánh đúng phân phối dữ liệu thật.

Tham số `random_state=42` được cố định để kết quả chia dữ liệu có thể tái lập khi chạy lại notebook.

## 2. Quy tắc chống data leakage

Mọi bước resampling phải được thực hiện sau khi đã chia train/test. Các kỹ thuật như Random Oversampling, Random Undersampling, SMOTE và ADASYN chỉ được áp dụng trên train set.

Test set phải được giữ nguyên trong toàn bộ quá trình huấn luyện và chỉ được dùng để đánh giá cuối cùng. Không được oversample, undersample, SMOTE hoặc ADASYN trên test set.

Quy tắc này giúp tránh data leakage, tức là tránh việc thông tin từ tập test bị đưa vào quá trình huấn luyện thông qua mẫu sao chép hoặc mẫu tổng hợp.

## 3. Preprocessing

Cột `id` không được đưa vào training vì đây chỉ là mã định danh của mẫu dữ liệu, không mang ý nghĩa dự đoán.

Các feature của dataset WDBC đều là dữ liệu số. Pipeline tiền xử lý gồm:

- Kiểm tra missing values.
- Nếu có missing values, xử lý bằng `SimpleImputer(strategy="median")`.
- Chuẩn hóa feature bằng `StandardScaler`.

Imputer và scaler chỉ được fit trên train set. Sau đó, cùng bộ biến đổi này được dùng để transform train set và test set. Không fit preprocessing trên toàn bộ dataset để tránh data leakage.

## 4. Model

Để so sánh công bằng giữa các phương pháp xử lý mất cân bằng dữ liệu, các cấu hình thí nghiệm sử dụng cùng một mô hình phân loại cơ sở:

```python
LogisticRegression(max_iter=1000, random_state=42)
```

Việc giữ nguyên mô hình giúp sự khác biệt về kết quả chủ yếu đến từ kỹ thuật xử lý mất cân bằng dữ liệu, thay vì do thay đổi thuật toán phân loại.

## 5. Metric đánh giá

Vì dataset bị mất cân bằng lớp, accuracy không được dùng làm metric chính. Các metric đánh giá bắt buộc gồm:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC hoặc Average Precision
- Confusion Matrix

Trong dataset WDBC, lớp `M` (Malignant) được xem là minority/positive class. Vì vậy, Precision, Recall và F1-score cần tập trung vào khả năng nhận diện đúng lớp `M`, thay vì chỉ nhìn vào độ chính xác tổng thể.

## 6. Tóm tắt quy định

| Thành phần | Quy định |
|---|---|
| Split | Train 80%, test 20% |
| Stratify | Có, dùng `stratify=y` |
| Random state | `42` |
| Resampling | Chỉ áp dụng trên train set |
| Test set | Giữ nguyên, chỉ dùng để đánh giá cuối cùng |
| Preprocessing | Median imputer và `StandardScaler` |
| Model | `LogisticRegression(max_iter=1000, random_state=42)` |
| Metric | Precision, Recall, F1-score, ROC-AUC, PR-AUC/Average Precision, Confusion Matrix |
