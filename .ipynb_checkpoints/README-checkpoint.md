# Báo cáo Máy Học

### Nhóm 17: Trần Công Tuấn, Bảo Hoàng Thiên Phúc, Trần Nhật Quang Huy, Trần Thanh Phúc 

# Dự đoán đội chiến thắng trong trận đấu giải Ngoại Hạng Anh - Premier League

## 1. Giới thiệu

Bóng đá là bộ môn thể thao phổ biến nhất trên thế giới. Một số quốc gia có nhiều đội bóng thi đấu ở giải đấu vô địch khu vực và cấp quốc gia. Trong số các giải vô địch quốc gia trên thế giới thì đề tài này chủ yếu tập trung vào giải Ngoại Hạng Anh (Premier League), nơi giải đấu có nhiều người xem nhất trên thế giới.

### 1.1. Mục tiêu

- Xây dựng được mô hình dự đoán một trận đấu bóng đá và so sánh độ chính xác giữa các phương pháp phân lớp (đạt độ chính xác hơn 50% và duy trì được 60%).
- Giải thích được các yếu tố ảnh hưởng đến kết quả của các trận đấu diễn ra.

### 1.2. Giới hạn

- Dữ liệu (dataset) được lấy từ trang [fbref](https://fbref.com/en/comps/9/Premier-League-Stats)
- Thuật toán được sử dụng để xây dựng mô hình gồm có: RandomForest, SVM, kNN và Neural Network

## 2. Chuẩn bị dữ liệu

### 2.1. Thu thập dữ liệu

Dữ liệu được tổng hợp ở bảng: "Regular season" [ở đây](https://fbref.com/en/comps/9/Premier-League-Stats) và chi tiết dữ liệu cho từng đội ở bảng "Shotting"

<img src="Figures/scrap1.png">
<em>Dữ liệu ở bảng Regular season</em>

<img src="Figures/scrap2.png">
<em>Dữ liệu ở bảng Shooting của từng đội</em>

[Chi tiết code tại đây](scraping.ipynb)

### 2.2. Sơ chế dữ liệu (Data Wrangling)

Dữ liệu ta tổng hợp được có các đặc trưng:

- date: Ngày diễn ra trận đấu
- time: Thời gian diễn ra trận đấu
- comp: Giải đấu
- round: Vòng đấu
- day: Thứ trong tuần
- venue: Địa điểm tổ chức trận đấu
- result: Kết quả trận đấu
- gf: Số bàn thắng của đội chủ nhà
- ga: Số bàn thua của đội chủ nhà
- opponent: Đội đối thủ
- xg: Xếp hạng xG (Expected Goals) của đội chủ nhà
- xga: Xếp hạng xG (Expected Goals) của đội đối thủ
- poss: Tỷ lệ kiểm soát bóng của đội chủ nhà
- attendance: Số lượng khán giả có mặt
- captain: Đội trưởng của đội chủ nhà
- formation: Hệ thống chiến thuật của đội chủ nhà
- referee: Trọng tài
- match report: Báo cáo trận đấu
- notes: Ghi chú
- sh: Số cú sút của đội chủ nhà
- sot: Số cú sút trúng đích của đội chủ nhà
- dist: Tổng quãng đường chạy của đội chủ nhà
- fk: Số lượt sút phạt của đội chủ nhà
- pk: Số lượt sút penalty của đội chủ nhà
- pkatt: Số lượt sút penalty thực hiện của đội chủ nhà
- season: Mùa giải
- team: Tên đội chủ nhà

Các cột dữ liệu trên cung cấp thông tin về kết quả, số liệu thống kê và các yếu tố liên quan đến trận đấu và đội bóng chủ nhà.

Các cột như "gf", "ga", "xg", "xga", "poss", "sh", "sot", "dist", "fk", "pk", "pkatt" là các chỉ số thể hiện khả năng tấn công và phòng thủ của đội bóng chủ nhà trong trận đấu.

Thông qua việc phân tích và xử lý dữ liệu này, ta có thể trích chọn đặc trưng (Feature Selection) để dự đoán kết quả và đội chiến thắng trong các trận đấu giải Ngoại Hạng Anh.
Xóa các đặc trưng không cần thiết

Có 2 đặc trưng không cần thiết mà lúc thu thập dữ liệu ta đã lấy

```python
del matches["comp"]
del matches["notes"]
```

Chuyển đổi dữ liệu đúng định dạng

```python
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
```

[Chi tiết code tại đây](data_clean_and_preprocessing.ipynb)

### 2.3. Trực quan hóa dữ liệu (Data Visualization)

<img src="Data Visualization\6.png">
<em>Figure 1. Phân bố kết quả trận đấu</em>

<img src="Data Visualization\13.png">
<em>Figure 2. Biểu đồ cho các đặc trưng</em>

<img src="Data Visualization\12.png" alt="Figure 1">
<em>Figure 3. Tổng số bàn thắng của từng đội theo mùa giải</em>

<img src="Data Visualization\10.png" alt="Figure 2">
<em>Figure 4. Phân bố sút phạt penalty</em>

<img src="Data Visualization\3.png" alt="Figure 3">
<em>Figure 5.Biểu đồ boxplot cho xếp hạng xG của đội chủ nhà theo giải đấu</em>

<img src="Data Visualization\1.png" alt="Figure 1">
<em>Figure 6. Trực quan hóa phân phối của số bàn thắng của đội chủ nhà</em>

<img src="Data Visualization\8.png" alt="Figure 2">
<em>Figure 7. Biểu đồ tương quan giữa xếp hàng xg và số bàn thắng</em>

<img src="Data Visualization\9.png" alt="Figure 2">
<em>Figure 8. Biểu đồ tương quan giữa kiểm soát bóng và số bàn thắng</em>

<img src="Data Visualization\11.png" alt="Figure 2">
<em>Figure 9. Biểu đồ tương quan giữa xếp hàng xg và tỷ lệ kiểm soát bóng</em>

<img src="Data Visualization\2.png" alt="Figure 2">
<em>Figure 10. Biểu đồ tương quan giữa tỷ lệ kiểm soát bóng và số cú sút của đội chủ nhà</em>

<img src="Data Visualization\7.png" alt="Figure 2">
<em>Figure 11. Biểu đồ tương quan giữa tỷ lệ kiểm soát bóng và số cú sút của đội chủ nhà</em>

<img src="Data Visualization\newplot.png" alt="Figure 4">
<em>Figure 12. Biểu đồ tương quan giữa số lượng khán giả có mặt và số bàn thắng đội nhà</em>

<img src="Data Visualization\4.png" alt="Figure 5">
<em>Figure 13. Biểu đồ heatmap để hiển thị ma trận tương quan giữa các biến</em>

[Chi tiết code tại đây](Data%20Visualization/data_visualization.ipynb)

### 2.4. Trích chọn đặc trưng (Feature Selection)

Thực hiện trích chọn đặc trưng, sử dụng SelectKBest để chọn K đặc trưng quan trọng nhất

<img src="Data Visualization\5.png" alt="Figure 5">

<em>Figure 14. Biểu đồ các đặc trưng được chọn</em>

## 3. Chọn mô hình và huấn luyện

Đồ án sẽ xây dựng các mô hình giúp dự đoán kết quả trận đấu.

Lấy dữ liệu 'matches 2019-2023.xls'
Phân chia dữ liệu thành 2 tập: Training Set ( date<'2022-01-01') và Test Set (date >'2022-01-01').

Khởi tạo mô hình, nhận thuật toán phân lớp thông qua tham số clf (classifier) của hàm.

Đánh giá mô hình bằng Kiểm chứng chéo (Cross Validation) trên Training Set

Đánh giá 4 hệ số: Accuracy (Độ chính xác tổng quát), Precision (Độ chính xác), Recall (Độ nhạy), F1.

Vẽ Confusion Matrix cho kết quả dự đoán của mô hình trên Test Set.

Các thuật toán lựa chọn và sử dụng từ thư viện scikit-learn. Tất cả mô hình được tối ưu hóa bằng cách sử dụng grid search

- SVM
- K-Nearest Neighbour
- RandomForest
- Neural Network

### 3.1. Đánh giá nhanh độ chính xác của mô hình

Ta kiểm tra nhanh độ chính xác của mô hình bằng cách sử dụng thuật toán RandomForest

```
accuracy score:  0.600739371534196
precision score:  0.48736462093862815
recall score:  0.3176470588235294
f1 score:  0.3846153846153846
```

| predicted |  0  |  1  |
| --------- | :-: | :-: |
| actual    |     |     |
| 0         | 515 | 142 |
| 1         | 290 | 135 |

Nhận thấy accuracy score khá ổn tuy nhiên precision score chỉ xấp xỉ 0.48, số lần dự đoán đúng thực thế khá thấp vì thế cần tiến hành cải thiện độ chính xác cho mô hình

### 3.2. Tối ưu tập dữ liệu

Ta tính mức trung bình hiệu suất của đội bóng qua các trận đấu là các đặc trưng: ghi được bao nhiêu bàn thắng, số cú sút
Tiến hành gom dữ liệu theo từng đội bóng và tính toán trung bình luân phiên dữ liệu của 3 tuần trước đó để chuyển dữ liệu đó vào tuần thứ tư

```
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])
```

<img src="Figures/data_after_preprocessing.png" alt="Figure 5">

## 4. Xây dựng mô hình dự đoán

### 4.1. Random Forest

Các chỉ số của mô hình sau khi huấn luyện:

```
Confusion Matrix:
[[169  96]
 [ 77 100]]
Accuracy Score: 0.665
Precision Score: 0.698
Recall Score: 0.288
F1 Score: 0.408

Cross-Validation Accuracy Score: 64.8%
```

<img src="rf/figures/1.png" alt="Figure 12">
<em>Figure 12. Biểu đồ Confusion Matrix của Random Forest</em>

<img src="rf/figures/2.png" alt="Figure 13">
<em>Figure 13. Learning Curve của Random Forest</em>

<img src="rf/figures/3.png" alt="Figure 14">
<em>Figure 14. Feature Importance của Random Forest</em>

[Chi tiết code tại đây](rf/predict_rf.ipynb)

### 4.2. Support Vector Machine

Các chỉ số của mô hình sau khi huấn luyện:

```
Confusion Matrix:
[[241  24]
 [146  31]]
Accuracy Score: 0.626
Precision Score: 0.700
Recall Score: 0.118
F1 Score: 0.202

Cross-Validation Accuracy Score: 63.2%
```

<img src="svm/figures/1.png" alt="Figure 15">
<em>Figure 15. Biểu đồ Confusion Matrix của SVM</em>

<img src="svm/figures/2.png" alt="Figure 16">
<em>Figure 16. Learning Curve của SVM</em>

[Chi tiết code tại đây](svm/predict_svm.ipynb)

### 4.3. K-Nearest Neighbors

Các chỉ số của mô hình sau khi huấn luyện:

```
Confusion Matrix:
[[196  69]
 [124  53]]
Accuracy Score: 0.633
Precision Score: 0.584
Recall Score: 0.294
F1 Score: 0.391

Cross-Validation Accuracy Score: 62.9%
```

<img src="knn/figures/1.png" alt="Figure 17">
<em>Figure 17. Biểu đồ Confusion Matrix của KNN</em>

<img src="knn/figures/2.png" alt="Figure 18">
<em>Figure 18. Learning Curve của KNN</em>

[Chi tiết code tại đây](knn/predict_knn.ipynb)

### 4.4. Neural Network

Các chỉ số của mô hình ban đầu:

```
Confusion Matrix:
[[169  96]
 [ 77 100]]
Accuracy Score: 0.609
Precision Score: 0.510
Recall Score: 0.565
F1 Score: 0.536
```

Sau khi tối ưu hóa với các cải tiến:
- Chuẩn hóa dữ liệu với StandardScaler
- Random Oversampling để cân bằng dữ liệu
- Tăng số lần lặp max_iter lên 1000
- Thêm early stopping để tránh overfitting
- Thêm learning rate và activation vào grid search

Các chỉ số của mô hình sau khi tối ưu:

```
Confusion Matrix:
[[241  24]
 [146  31]]
Accuracy Score: 0.615
Precision Score: 0.564
Recall Score: 0.175
F1 Score: 0.267

Cross-Validation Accuracy Score: 64.4%
```

<img src="mlp/figures/1.png" alt="Figure 19">
<em>Figure 19. Biểu đồ Confusion Matrix của Neural Network</em>

<img src="mlp/figures/2.png" alt="Figure 20">
<em>Figure 20. Heatmap thể hiện độ chính xác với các kích thước tầng ẩn khác nhau</em>

[Chi tiết code tại đây](mlp/predict_mlp.ipynb)

## 5. Đánh giá mô hình

### 5.1. Đánh giá giữa các mô hình

Dưới đây là biểu đồ so sánh các hệ số đánh giá giữa các mô hình máy học đã được xây dựng để giải quyết bài toán ban đầu. Gồm có 4 giá trị: Accuracy (Độ chính xác tổng quát), Precision (Độ chính xác), Recall (Độ nhạy), F1.
<img src="Figures/1.png" alt="Figure 17">
<em>Figure 26. So sánh các hệ số giữa các mô hình</em>

<img src="Figures/2.png" alt="Figure 17">
<em>Figure 27. So sánh Cross-Validation Accuracy Score giữa các mô hình</em>

[Chi tiết code tại đây](Evaluation.ipynb)

<!-- Accuracy: Mô hình Random Forest có điểm số cao nhất với khoảng 0.665, tiếp theo là SVM (0.626), kNN (0.633) và Neural Network (0.615). Tuy nhiên, sự khác biệt giữa các mô hình không lớn.

Precision: Mô hình SVM có độ chính xác (precision) cao nhất với khoảng 0.7, tiếp theo là Random Forest (0.698), kNN (0.584) và Neural Network (0.52).

Recall: Mô hình Neural Network có khả năng phát hiện các dữ liệu positive (recall) cao nhất với khoảng 0.514, tiếp theo là kNN (0.294), Random Forest (0.288) và SVM (0.118).

F1-score: Mô hình Random Forest có F1-score cao nhất với khoảng 0.408, tiếp theo là Neural Network (0.517), kNN (0.391) và SVM (0.202). -->

<!-- Random Forest có vẻ có hiệu suất tốt nhất với các chỉ số Accuracy và F1-score cao nhất trong số các mô hình. Do đó, nếu bạn muốn chọn một mô hình duy nhất, có thể xem xét mô hình Random Forest như một lựa chọn tiềm năng. Tuy nhiên, cần xem xét kỹ càng và điều chỉnh các siêu tham số của mô hình để đạt hiệu suất tốt nhất trong bối cảnh cụ thể của bài toán của bạn. -->
<!--
Recall: Mô hình Neural Network có giá trị Recall cao nhất với khoảng 0.514, tiếp theo là kNN (0.293), Random Forest (0.242) và SVM (0.118). Recall đo lường khả năng của mô hình trong tìm ra các trường hợp Positive. Mô hình Neural Network có khả năng tìm ra các trường hợp Positive tốt nhất, trong khi SVM có khả năng tìm ra các trường hợp Positive thấp nhất trong số các mô hình.

F1-score: Mô hình Neural Network có giá trị F1-score cao nhất với khoảng 0.517, tiếp theo là Random Forest (0.408), kNN (0.391) và SVM (0.202). F1-score là một chỉ số kết hợp giữa Precision và Recall, đo lường sự cân bằng giữa độ chính xác và khả năng tìm ra các trường hợp Positive. Mô hình Neural Network đạt được một sự cân bằng tốt giữa Precision và Recall, trong khi SVM có F1-score thấp nhất trong số các mô hình. -->

### 5.2. Lựa chọn mô hình

Nếu độ chính xác là yếu tố quan trọng nhất: thì có thể chọn mô hình Random Forest, vì nó có độ chính xác tương đối cao.

Nếu khả năng phát hiện và phân loại đúng các mẫu là ưu tiên: có thể chọn Neural Network hoặc kNN, vì cả hai mô hình này có recall và F1-score cao hơn so với các mô hình khác.

Nếu cân bằng giữa precision và recall: Trong trường hợp này, cả Neural Network và kNN có F1-score tương đối cao.

## 6. Kết luận

Với dữ liệu được chuẩn bị, sau khi thực hiện tiền xử lý dữ liệu, tối ưu lại dữ liệu thì đã xây dựng được mô hình dự đoán duy trì độ chính xác trung bình là 60% và có thể đạt tới 70%

Tuy nhiên với sự khắc nghiệt của bóng đá giải Ngoại Hạng Anh nơi mà bất cứ kịch bản nào cũng có thể xảy ra thì 1 đội bóng được đánh giá thấp hơn cũng có thể có được chiến thắng. Đó cũng chính là vấn đề mà mô hình dự đoán này gặp phải.

Những việc sẽ được làm tiếp theo để cải thiện mô hình dự đoán

- Thu thập thêm nhiều dữ liệu của nhiều mùa trước đó
- Dùng thêm nhiều phương pháp khác nhau để giải quyết(PCA, LDA)
- Tìm thêm các đặc trưng ảnh hưởng đến kết quả dự đoán

## 5. Đánh giá mô hình ===========================================(New)========================================================

### 5.1. Đánh giá giữa các mô hình

Dưới đây là biểu đồ so sánh các hệ số đánh giá giữa các mô hình máy học đã được xây dựng để giải quyết bài toán ban đầu. Gồm có 4 giá trị: Accuracy (Độ chính xác tổng quát), Precision (Độ chính xác), Recall (Độ nhạy), F1.

<img src="Figures/1.png" alt="Figure 17">
<em>Figure 26. So sánh các hệ số giữa các mô hình</em>

<img src="Figures/2.png" alt="Figure 17">
<em>Figure 27. So sánh Cross-Validation Accuracy Score giữa các mô hình</em>

[Chi tiết code tại đây](Evaluation.ipynb)

### 5.2. Lựa chọn mô hình

Dựa trên kết quả đánh giá từ các mô hình:

1. Random Forest:
- Accuracy: 66.5%
- Cross-validation accuracy: 64.8%
- Ưu điểm: Độ chính xác cao nhất, ổn định qua cross-validation
- Feature importance cho phép hiểu rõ tầm quan trọng của từng đặc trưng

2. Neural Network (MLP):
- Accuracy: 61.5%
- Cross-validation accuracy: 64.4%
- Cải thiện: Thêm chuẩn hóa dữ liệu, random oversampling và early stopping
- Khả năng học tốt qua các epoch

3. SVM:
- Accuracy: 62.6%
- Cross-validation accuracy: 63.2%
- Hiệu quả với dữ liệu phi tuyến tính
- Thời gian huấn luyện lâu hơn các mô hình khác

4. KNN:
- Accuracy: 63.3%
- Cross-validation accuracy: 62.9%
- Đơn giản, dễ hiểu
- Nhạy cảm với nhiễu và outliers

Lựa chọn mô hình phù hợp:

1. Nếu ưu tiên độ chính xác cao nhất: Chọn Random Forest với accuracy 66.5% và khả năng giải thích được tầm quan trọng của các đặc trưng.

2. Nếu cần mô hình có khả năng học phức tạp: Neural Network với các cải tiến đã thêm vào giúp mô hình học tốt hơn.

3. Nếu cần mô hình đơn giản, dễ triển khai: KNN là lựa chọn phù hợp với accuracy khá tốt 63.3%.

## 6. Kết luận

Với dữ liệu được chuẩn bị, sau khi thực hiện tiền xử lý dữ liệu và tối ưu các mô hình, chúng tôi đã xây dựng được hệ thống dự đoán với độ chính xác trung bình trên 60% và có thể đạt tới 66.5% với Random Forest.

Các cải tiến đã thực hiện:
- Chuẩn hóa dữ liệu với StandardScaler
- Random Oversampling để cân bằng dữ liệu
- Grid Search để tìm tham số tối ưu
- Early stopping để tránh overfitting
- Feature engineering và selection

Tuy nhiên với sự khắc nghiệt của bóng đá giải Ngoại Hạng Anh nơi mà bất cứ kịch bản nào cũng có thể xảy ra thì 1 đội bóng được đánh giá thấp hơn cũng có thể có được chiến thắng. Đó cũng chính là vấn đề mà mô hình dự đoán này gặp phải.

Những việc sẽ được làm tiếp theo để cải thiện mô hình dự đoán:

- Thu thập thêm nhiều dữ liệu của nhiều mùa trước đó
- Áp dụng thêm các phương pháp khác như PCA, LDA để giảm chiều dữ liệu
- Tìm thêm các đặc trưng ảnh hưởng đến kết quả dự đoán
- Thử nghiệm các kiến trúc deep learning phức tạp hơn
- Kết hợp nhiều mô hình (ensemble learning)

