**Thống nhất Metric Đánh giá Mô hình**  
Đối với bài toán phân loại nhị phân trên tập dữ liệu mất cân bằng (Breast Cancer), nhóm sẽ không sử dụng Accuracy để tránh sai lệch. Các metric bắt buộc được chốt để đánh giá tất cả các phương pháp bao gồm:

- Precision: Tỷ lệ dự đoán đúng trong tổng số các ca được dự đoán là dương tính.
- Recall: Tỷ lệ phát hiện đúng trong tổng số các ca thực sự dương tính.
- F1-score: Trung bình điều hòa giữa Precision và Recall.
- ROC-AUC: Đánh giá khả năng phân tách giữa hai lớp của mô hình ở nhiều ngưỡng khác nhau.
- PR-AUC (Average Precision): Metric quan trọng nhất cho dữ liệu mất cân bằng, tập trung vào hiệu suất trên lớp thiểu số (minority class).
- Confusion Matrix: Ma trận nhầm lẫn để nhìn rõ số lượng True Positive, False Positive, True Negative, False Negative.

`from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix`

`def calculate_group_metrics(y_true, y_pred, y_pred_proba):`  
 `metrics = {`  
 `'Precision': precision_score(y_true, y_pred, pos_label=1),`  
 `'Recall': recall_score(y_true, y_pred, pos_label=1),`  
 `'F1-score': f1_score(y_true, y_pred, pos_label=1),`

        `'ROC-AUC': roc_auc_score(y_true, y_pred_proba),`
        `'PR-AUC': average_precision_score(y_true, y_pred_proba, pos_label=1)`
    `}`
    `cm = confusion_matrix(y_true, y_pred)`

    `return metrics, cm`

**Thống nhất Schema Bảng Kết Quả**

Nhằm đảm bảo việc tổng hợp kết quả từ các phương pháp resampling được đồng nhất, nhóm quy định định dạng file lưu kết quả cuối cùng như sau:

- **Tên file chuẩn:** metrics_comparison.csv
- **Cấu trúc các cột (Schema):** Cần tuân thủ đúng thứ tự và tên cột để Code tổng hợp không bị lỗi.
  1. Method: Tên phương pháp chạy (Quy chuẩn các giá trị: _Baseline, Random Oversampling, Random Undersampling, SMOTE, ADASYN_).
  2. Precision: Ghi nhận metric Precision của lớp M.
  3. Recall: Ghi nhận metric Recall của lớp M.
  4. F1-score: Ghi nhận metric F1-score của lớp M.
  5. ROC-AUC: Ghi nhận metric ROC-AUC.
  6. PR-AUC: Ghi nhận metric PR-AUC (Average Precision).
