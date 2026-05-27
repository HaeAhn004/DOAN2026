## PHÂN TÍCH CÁC YẾU TỐ ẢNH HƯỞNG VÀ ỨNG DỤNG HỌC MÁY TRONG DỰ ĐOÁN HÀNH VI KHÁM SỨC KHỎE CHỦ ĐỘNG

# TỔNG QUAN DỰ ÁN

Mục tiêu:
   - Nhận diện các rào cản (tâm lý, kinh tế, thời gian, chất lượng dịch vụ) khiến người dân trì hoãn khám sức khỏe định kỳ
    
   - Xây dựng mô hình Machine Learning dự đoán hành vi khám chủ động
    
   - Đề xuất giải pháp ứng dụng công nghệ (Telemedicine App tích hợp AI)

Bài toán: Binary Classification

   - Nhãn 0: Khám chủ động/thường xuyên
    
   - Nhãn 1: Trì hoãn khám/Nguy cơ cao

# THÔNG TIN DỮ LIỆU

Nguồn dữ liệu: Vietnam Health Survey (Carnegie Mellon University)

Số lượng mẫu: 2.068 rows

Số lượng biến: 50 columns (sau làm sạch: 46 columns)

Các nhóm biến chính:
   - Nhân khẩu học: Age, Sex, Edu, Jobstt, Income
    
   - Thể chất: height, weight, BMI
    
   - Hành vi y tế: RecPerExam, ReaExam, SuitFreq
    
   - Rào cản: Wsttime, Wstmon, DiscDisease

   - Chất lượng dịch vụ: Tangibles, Reliability, Respon, Assurance, Empathy

   - Thói quen: SuitExer, EvalExer, FlwHealth, Habit

   - Công nghệ: UseIT, AfterIT

Phân bố biến mục tiêu:
   - Nhãn 0 (Khám chủ động): 1.277 mẫu (61.8%)
    
   - Nhãn 1 (Trì hoãn khám): 791 mẫu (38.2%)


# CÀI ĐẶT THƯ VIỆN

  requirements.txt:
    pandas>=1.3.0
    numpy>=1.21.0
    scikit-learn>=1.0.0
    xgboost>=1.5.0
    lightgbm>=3.3.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    imbalanced-learn>=0.8.0
    joblib>=1.1.0

# TIỀN XỬ LÝ DỮ LIỆU


    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
 
    def load_and_preprocess_data(filepath):
    "Đọc và tiền xử lý dữ liệu"
    df = pd.read_csv(filepath)
    
    # Loại bỏ các biến định danh/nhiễu
    cols_to_drop = ['id', 'date', 'Unnamed']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Xử lý giá trị "unknow" - giữ nguyên như một nhãn độc lập
    # (phản ánh hành vi thờ ơ với y tế)
    
    # Label Encoding cho các biến phân loại
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Xác định biến mục tiêu: Target_Risk
    # less12, b1224 -> 0 (chủ động)
    # g24, unknow -> 1 (trì hoãn)
    
    return df, label_encoders

    def split_data(df, target_col, test_size=0.2, random_state=42):
    """Phân tách tập train/test"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Tập huấn luyện: {X_train.shape}")
    print(f"Tập kiểm thử: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# KẾT QUẢ PHÂN TÍCH EDA (11 BƯỚC) - TÓM TẮT

    eda_insights = {
    "Bước_1_Nhân_khẩu_học": {
        "nữ_giới": "64.8% (1.340/2.068)",
        "tuổi_trung_bình": "29.17",
        "học_vấn_cao": "1.383 Đại học/CĐ, 127 Sau ĐH",
        "nghề_nghiệp_chính": "1.123 người đi làm ổn định, 548 sinh viên"
    },
    "Bước_2_Thể_chất_BMI": {
        "BMI_trung_bình": "20.84 (vùng tiêu chuẩn)",
        "nam_giới_BMI_cao_hơn": "nữ giới ở mọi độ tuổi",
        "nguy_cơ_cao": "nam trung niên (40-49) BMI cao nhất"
    },
    "Bước_3_Tần_suất_Lý_do_khám": {
        "khám_tự_nguyện": "25% (514/2.068)",
        "khám_do_triệu_chứng": "730 người",
        "khám_do_yêu_cầu": "726 người",
        "nghịch_lý": "50% khám dưới 12 tháng nhưng 1/2 trong số đó là do bắt buộc"
    },
    "Bước_4_Khoảng_trống_nhận_thức": {
        "ưu_tiên_sức_khỏe": "81% (1.675 người)",
        "bỏ_khám_trong_nhóm_này": "36% (600+ người - g24 hoặc unknow)"
    },
    "Bước_5_Rào_cản": {
        "thời_gian": "rào cản phổ quát nhất (47-49% người đi làm/nội trợ)",
        "tiền_bạc": "45.6% lao động bấp bênh, 61.5% sinh viên",
        "sợ_phát_hiện_bệnh": "33.9% sinh viên (cao nhất tập)"
    },
    "Bước_6_Mất_niềm_tin": {
        "tỷ_lệ_mất_tin": "26.8% (554 người)",
        "nguyên_nhân_chính": "Respon (chờ lâu) và Empathy (thiếu thấu cảm)",
        "yếu_tố_quan_trọng_nhất": "Assurance (chuyên môn) và Reliability"
    },
    "Bước_7_Môi_trường_Gia_đình": {
        "có_tủ_thuốc_giúp_tăng_khám": "55.72% less12 vs 38.96%",
        "có_thói_quen_gia_đình": "tỷ lệ khám lên 66.78% (cao nhất tập)"
    },
    "Bước_8_Lịch_sử_bệnh_lý": {
        "có_người_nhà_bệnh": "tỷ lệ khám giảm từ 54.91% xuống 46.56%",
        "nghịch_lý": "bệnh tật không tạo thức tỉnh, mà tạo chối bỏ"
    },
    "Bước_9_Hành_vi_khi_ốm": {
        "GenZ_tự_tra_Google": "30% ở nhóm 18-29 tuổi",
        "trên_40_tuổi_đi_khám_ngay": "62-64%"
    },
    "Bước_10_Kỳ_vọng_giá": {
        "mong_muốn": "giá <2 triệu, tần suất 6-12 tháng",
        "phân_khúc_đông_nhất": "giá med (1-2tr) và low (<1tr)"
    },
    "Bước_11_Sẵn_sàng_công_nghệ": {
        "sẵn_sàng_dùng_App": "77% (42% chắc chắn + 35% có thể)",
        "chuyển_đổi_nhóm_ngủ_đông": "38.55% sẽ đặt lịch nếu được cảnh báo"
    }
}

# HUẤN LUYỆN CÁC MÔ HÌNH

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV

# 1. Logistic Regression - Baseline
    def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# 2. XGBoost
    def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# 3. LightGBM
    def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return model

# 4. Random Forest với GridSearchCV (MÔ HÌNH TỐT NHẤT)
    def train_random_forest_optimized(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# ĐÁNH GIÁ MÔ HÌNH VỚI THRESHOLD TÙY CHỈNH

    def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """Đánh giá mô hình với ngưỡng phân lớp tùy chỉnh"""
    
    # Dự đoán xác suất
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Áp dụng threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Tính các chỉ số
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': auc,
        'threshold_used': threshold
    }

# KẾT QUẢ MÔ HÌNH RANDOM FOREST (THRESHOLD = 0.4)

    random_forest_results = {
    "best_params": {
        "n_estimators": 200,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_depth": 15
    },
    "threshold": 0.4,
    "classification_report": {
        "class_0": {"precision": 0.87, "recall": 0.64, "f1_score": 0.74, "support": 256},
        "class_1": {"precision": 0.59, "recall": 0.84, "f1_score": 0.69, "support": 158}
    },
    "accuracy": 0.72,
    "roc_auc": 0.7926,
    "confusion_matrix": {
        "true_negative": 164,
        "false_positive": 92,
        "false_negative": 25,
        "true_positive": 133
    }
}

# SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH
    model_comparison = {
    "Logistic_Regression": {
        "recall_class_1": "Thấp",
        "f1_class_1": "Thấp",
        "roc_auc": "Khá",
        "ghi_chú": "Baseline, chỉ bắt được quan hệ tuyến tính"
    },
    "LightGBM": {
        "recall_class_1": None,
        "f1_class_1": None,
        "roc_auc": None,
        "ghi_chú": "Underfitting - dừng chia nhánh sớm"
    },
    "XGBoost": {
        "recall_class_1": "~0.60",
        "f1_class_1": "~0.65",
        "roc_auc": "~0.72",
        "ghi_chú": "Strong Baseline phi tuyến"
    },
    "Random_Forest_Th0.4": {
        "recall_class_1": "0.84",
        "f1_class_1": "0.69",
        "roc_auc": "0.7926",
        "ghi_chú": "MÔ HÌNH ĐƯỢC CHỌN - Tối ưu cho y tế"
    }
}

# LÝ DO CHỌN RANDOM FOREST

    reasons_for_selection = {
    "1_Hiệu_năng_vượt_trội": "Recall = 0.84 - bắt đúng 84% người lười khám",
    "2_Triết_lý_y_tế": "Thà cảnh báo nhầm (FP) còn hơn bỏ sót (FN) - chi phí sai lầm thấp",
    "3_Chống_overfitting": "max_depth=15, min_samples_leaf=2 giúp khái quát hóa tốt",
    "4_Triển_khai_nhẹ": "Đủ nhẹ để đóng gói vào Web/App, ít tài nguyên"

# ỨNG DỤNG THỰC TẾ - TELEMEDICINE APP TÍCH HỢP AI


    
    def __init__(self, model, threshold=0.4):
        self.model = model
        self.threshold = threshold
    
    def predict_risk(self, patient_features):
        """Chấm điểm rủi ro cho bệnh nhân mới"""
        proba = self.model.predict_proba([patient_features])[0][1]
        risk_label = 1 if proba >= self.threshold else 0
        return {"risk_probability": proba, "risk_label": risk_label}
    
    def generate_nudge(self, patient_profile, risk_score):
        """Tạo cú hích phù hợp với từng rào cản"""
        
        # Cú hích Tài chính (Sinh viên/Lao động tự do)
        if patient_profile['jobstt'] in ['student', 'unstable'] and risk_score > 0.5:
            return {
                "type": "financial",
                "message": "🎁 Voucher giảm giá 30% gói khám cơ bản (chỉ 690k)!",
                "action": "book_now"
            }
        
        # Cú hích Thời gian (Người đi làm ổn định)
        if patient_profile['jobstt'] == 'stable' and patient_profile.get('wsttime', 0) > 0.5:
            return {
                "type": "time",
                "message": "⏰ Đặt lịch khám ngoài giờ (Thứ 7/CN) - Cam kết không chờ đợi!",
                "action": "book_weekend"
            }
        
        # Cú hích Tâm lý (Nhóm sợ phát hiện bệnh)
        if patient_profile.get('discdisease', 0) == 1:
            return {
                "type": "psychological",
                "message": "📚 Phát hiện sớm là cơ hội điều trị tốt nhất. Đọc cẩm nang sức khỏe!",
                "action": "read_guide"
            }
        
        # Cú hích mặc định
        return {
            "type": "default",
            "message": "🏥 Đã hơn 2 năm bạn chưa khám tổng quát. Đặt lịch ngay hôm nay!",
            "action": "book_now"
        }
    
    def symptom_checker(self, symptoms_text):
        """Tính năng AI Symptom Checker - thu phục nhóm Bác sĩ Google"""
        # Xử lý triệu chứng, đưa ra phán đoán sơ bộ
        preliminary_diagnosis = self.analyze_symptoms(symptoms_text)
        
        # Kêu gọi hành động
        return {
            "preliminary_result": preliminary_diagnosis,
            "call_to_action": "⚠️ Triệu chứng của bạn cần bác sĩ xác nhận. Đặt tư vấn online 5 phút!",
            "action": "telemedicine_consult"
        }
    
    def analyze_symptoms(self, symptoms_text):
        """Phân tích triệu chứng (placeholder)"""
        # Trong thực tế, đây sẽ là một mô hình NLP xử lý text
        return {"risk_level": "medium", "suggested_specialty": "Internal Medicine"}

# KIẾN TRÚC HỆ THỐNG ĐỀ XUẤT


    system_architecture = {
    "1_Backend_API": "NestJS/Node.js - Quản lý người dùng, lịch hẹn",
    "2_AI_Service": "Python FastAPI - Load model Random Forest, dự đoán rủi ro",
    "3_Frontend_App": "React Native/Next.js - Giao diện người dùng, push notification",
    "4_Database": "MongoDB/PostgreSQL - Lưu profile bệnh nhân, lịch sử dự đoán",
    "5_Infrastructure": "Docker + Docker Compose - Đóng gói và triển khai"

# HẠN CHẾ CỦA ĐỒ ÁN

    limitations = {
    "1_Thiên_lệch_mẫu": "Dữ liệu tập trung vào người trẻ (tuổi TB 29), tri thức đô thị",
    "2_Dữ_liệu_tĩnh": "Cross-sectional data - chưa theo dõi thay đổi hành vi theo thời gian",
    "3_Chưa_tích_hợp_IoT": "Chưa đồng bộ dữ liệu từ thiết bị đeo thông minh",
    "4_Phạm_vi_địa_lý": "Chưa khảo sát người dân vùng nông thôn, vùng sâu vùng xa"

# HƯỚNG PHÁT TRIỂN TƯƠNG LAI

    future_directions = {
    "1_Mở_rộng_dữ_liệu": "Khảo sát thêm người cao tuổi, nông thôn, thu nhập thấp",
    "2_Dashboard_PowerBI": "Giám sát thời gian thực tỷ lệ khám định kỳ",
    "3_Tích_hợp_Wearables": "Đồng bộ dữ liệu từ Apple Watch, vòng tay thông minh",
    "4_A/B_Testing_Nudge": "Thử nghiệm các loại cú hích khác nhau để tối ưu chuyển đổi",
    "5_Mở_rộng_mô_hình": "Phát triển thành bài toán multi-class (dự đoán loại hình khám bệnh)"

# KẾT LUẬN

Đồ án đã thành công trong việc:
    1. Xác định 11 nhóm yếu tố ảnh hưởng đến hành vi khám sức khỏe chủ động,
       bao gồm các rào cản tâm lý, kinh tế, thời gian và chất lượng dịch vụ.
    
   2. Xây dựng mô hình Random Forest với Recall = 0.84 cho nhóm trì hoãn khám,
       vượt trội so với các thuật toán khác, phù hợp với triết lý y tế
       "thà cảnh báo nhầm còn hơn bỏ sót".
    
   3. Đề xuất kiến trúc Telemedicine App tích hợp AI với cơ chế "cú hích" (Nudge)
       được cá nhân hóa theo từng rào cản, và tính năng AI Symptom Checker
       để thu phục nhóm "Bác sĩ Google".
    
   4. Đưa ra các khuyến nghị chiến lược cho phòng khám, bệnh viện về
       tái cấu trúc sản phẩm, chính sách trợ giá, và phương thức truyền thông y khoa.


# CHẠY HỆ THỐNG (MAIN)


    if __name__ == "__main__":
    print("=" * 80)
    print("ĐỒ ÁN TỐT NGHIỆP: DỰ ĐOÁN HÀNH VI KHÁM SỨC KHỎE CHỦ ĐỘNG")
    print("=" * 80)
    
    print("\n📊 KẾT QUẢ MÔ HÌNH TỐT NHẤT (Random Forest - Threshold 0.4):")
    print(f"   - Recall (nhóm trì hoãn): {random_forest_results['classification_report']['class_1']['recall']}")
    print(f"   - F1-Score (nhóm trì hoãn): {random_forest_results['classification_report']['class_1']['f1_score']}")
    print(f"   - ROC-AUC: {random_forest_results['roc_auc']}")
    
    print("\n🏥 ỨNG DỤNG THỰC TẾ:")
    print("   - Telemedicine App tích hợp AI dự đoán rủi ro")
    print("   - Cơ chế Nudge cá nhân hóa theo từng rào cản")
    print("   - AI Symptom Checker thu phục nhóm Bác sĩ Google")
    
    print("\n✅ KẾT LUẬN: Đồ án đã hoàn thành 4 mục tiêu đề ra, mô hình đạt hiệu suất cao,")
    print("   có thể triển khai thực tế để nâng cao tỷ lệ khám sức khỏe định kỳ trong cộng đồng.")
    print("=" * 80)
















  
