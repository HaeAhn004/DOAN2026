# Phân tích các yếu tố ảnh hưởng và ứng dụng Học máy (Machine Learning) trong dự đoán hành vi khám sức khỏe chủ động"
 
## CHƯƠNG 1: BỐI CẢNH VÀ VẤN ĐỀ NGHIÊN CỨU 
- Thực trạng: Người dân chủ yếu đi khám khi có triệu chứng, tỷ lệ khám định kỳ thấp dù nhận thức tốt.
- Khoảng trống dữ liệu: Hệ thống y tế chỉ lưu trữ hồ sơ bệnh nhân đi khám, thiếu dữ liệu hành vi của người chưa khám
- MỤC TIÊU:
  * Nhận diện rào cản (tâm lý, kinh tế, dịch vụ)
  * Xây dựng mô hình Machine Learning dự đoán hành vi khám chủ động
  * Đề xuất giải pháp cải thiện dịch vụ và ứng dụng công nghệ
- Dữ liệu 2,068 mẫu, 50 biến (nhân khẩu học, BMI, hành vi y tế,SERVQUAL,...)
- Bài toán: Binary classification - dự đoán "khám chủ động" (nhãn 0) và "trì hoãn khám" (nhãn 1).
      
## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & THỰC NGHIỆM

- Lý thuyết nền:
   * Health Behavior Theory (hành vi y tế).  
   * Mô hình SERVQUAL (chất lượng dịch vụ: Tangibles, Reliability, Empathy...).

- Kỹ thuật xử lý: Label Encoding, EDA, chọn đặc trưng.

- Chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

- Thuật toán sử dụng: 

  * Logistic Regression (baseline).

  * Random Forest, XGBoost, LightGBM.

  * Soft Voting Classifier (đề xuất nhưng không chọn cuối cùng).

  ## CHƯƠNG 3: PHƯƠNG PHÁP & QUY TRÌNH THỰC HIỆN

- Quy trình 7 bước: Dữ liệu thô → Làm sạch → EDA → Tiền xử lý → Modeling → Đánh giá → Ứng dụng.
- EDA được chia thành 11 phần trọng tâm, mỗi phần trả lời một câu hỏi hành vi:

 1 .  Chân dung nhân khẩu học, thể chất.
 
 2.  Tần suất & lý do khám (chỉ 25% tự nguyện).

 3.  Khoảng trống nhận thức – hành động (81% nói ưu tiên sức khỏe nhưng 36% bỏ khám).

 4.  Rào cản tiền – thời gian – sợ bệnh (sinh viên chịu áp lực cao nhất).

 5.  Tác động của BHYT & trợ cấp (có BHYT vẫn có 35% không đi khám).

 6.  Mất niềm tin dịch vụ (26,8% mất tin, nguyên nhân chính: chờ lâu + thiếu thấu cảm).

 7.  Hệ sinh thái y tế gia đình (gia đình có thói quen khám giúp tỷ lệ khám lên 66,8%).

 8.  Lịch sử bệnh lý: có người thân bệnh nặng làm giảm tỷ lệ khám.

 9.  Hành vi khi có triệu chứng: 30% Gen Z tự tra Google.

 10.  Kỳ vọng giá & tần suất: đa số muốn giá dưới 2 triệu, tần suất 6–12 tháng.

  11.  Sẵn sàng dùng App y tế: 77% có thể dùng, 38% nhóm bỏ khám lâu ngày sẽ đặt lịch nếu được cảnh báo.

   ## CHƯƠNG 4: MÔ HÌNH HÓA VÀ ĐÁNH GIÁ
- Tiền xử lý:
  * Giữ 45 đặc trưng, loại bỏ 4 cột nhiễu.
  * Target_Risk: 1.277 nhãn 0 (chủ động) – 791 nhãn 1 (trì hoãn).
  * Label Encoding cho biến phân loại, chia train/test = 80/20.
- Kết quả huấn luyện:
  * Logistic Regression (baseline) → hiệu suất thấp.
  * LightGBM → underfitting (dừng chia nhánh sớm).
  * XGBoost → khá, nhưng chưa tối ưu.
  * Random Forest (tối ưu) với threshold = 0.4:
      - Recall nhãn 1 (trì hoãn) = 0.84 – bắt đúng 84% người lười khám.
      - F1-score nhãn 1 = 0.69.
      - ROC-AUC = 0.7926.
  * Lý do chọn Random Forest:
    - Recall cao, phù hợp triết lý “thà cảnh báo nhầm hơn bỏ sót”.
    - Kiểm soát overfitting tốt.
    - Đủ nhẹ để triển khai thực tế.

   ## CHƯƠNG 5 : KẾT QUẢ VÀ BÀN LUẬN
- Tổng hợp phát hiện chính từ EDA:
    * Nghịch lý nhận thức – hành động.

    * Rào cản phân hóa theo nghề nghiệp, thế hệ.

    * Mất niềm tin vì chờ đợi và thiếu thấu cảm.

    * Thói quen gia đình là yếu tố dự báo mạnh nhất.

    * “Bác sĩ Google” phổ biến ở giới trẻ.

- Ứng dụng thực tế đề xuất:

    * Xây dựng Telemedicine App tích hợp AI.

    * Cá nhân hóa “cú hích” (Nudge) theo từng rào cản: giảm giá cho sinh viên, khám ngoài giờ cho người đi làm, tin tức sức khỏe cho nhóm sợ bệnh.

    * Tính năng AI Symptom Checker để thu phục nhóm tự tra Google.

- Hạn chế:

    * Dữ liệu thiên về người trẻ, tri thức đô thị.

    * Dữ liệu tĩnh (lát cắt ngang), chưa theo dõi thay đổi hành vi theo thời gian.

- Hướng phát triển:

    * Mở rộng khảo sát đa dạng đối tượng.

    * Xây dựng dashboard Power BI giám sát thời gian thực.

    * Tích hợp dữ liệu từ thiết bị đeo IoT.


   ## CHƯƠNG 6 : KẾT LUẬN & KHUYẾN NGHỊ
- Kết luận:

  * Đã xác định được các rào cản hành vi phức tạp.

  * Mô hình Random Forest đạt Recall 0.84, phù hợp triết lý y tế.

  * AI có thể chuyển hóa sự trì hoãn thành hành động qua App.

- Khuyến nghị thực tế:

  * Triển khai gói khám giá rẻ cho sinh viên, gói linh hoạt cho người đi làm.

  * Trợ giá cho người chăm sóc bệnh nhân dài hạn.

  * Thay đổi cách truyền thông y khoa theo nhóm đối tượng.

  * Tích hợp AI Symptom Checker để dẫn dắt người bệnh từ tự tra cứu sang đặt lịch khám.



















  
