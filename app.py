import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- 2. LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    model_path = 'Model/student_performance_model.joblib'
    scaler_path = 'Model/student_scaler.joblib'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return {'model': joblib.load(model_path), 'scaler': joblib.load(scaler_path)}
    return None

model_data = load_model()

# --- 3. MAPPING ---
MAP_COURSE = {
    "Biofuel Production Technologies": 1, "Animation and Multimedia Design": 2,
    "Social Service": 3, "Agronomy": 4, "Communication Design": 5,
    "Veterinary Nursing": 6, "Informatics Engineering": 7, "Equiniculture": 8,
    "Management": 9, "Social Service (evening attendance)": 10,
    "Tourism": 11, "Nursing": 12, "Oral Hygiene": 13, "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15, "Basic Education": 16, "Management (evening attendance)": 17
}

# --- 4. UI ---
st.title("🎓 Student Dropout Prediction")
st.markdown("### Jaya Jaya Institut Monitoring System")
st.info("Prediksi menggunakan 10 fitur utama berdasarkan Feature Importance model.")

if model_data is None:
    st.error("File model/scaler tidak ditemukan.")
else:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            course = st.selectbox("Program Studi", list(MAP_COURSE.keys()))
            tuition = st.radio("Uang Kuliah Lunas (Up to date)?", ["Ya", "Tidak"])
            age = st.slider("Usia Saat Pendaftaran", 17, 60, 20)
            adm_grade = st.number_input("Nilai Admission (Masuk)", 0.0, 200.0, 120.0)
            prev_grade = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 120.0)

        with col2:
            sem1_approved = st.number_input("Unit Semester 1 Disetujui", 0, 30, 5)
            sem1_grade = st.number_input("IP Semester 1", 0.0, 20.0, 12.0)
            sem2_eval = st.number_input("Jumlah Evaluasi Semester 2", 0, 50, 10)
            sem2_approved = st.number_input("Unit Semester 2 Disetujui", 0, 30, 5)
            sem2_grade = st.number_input("IP Semester 2", 0.0, 20.0, 12.0)

        submit = st.form_submit_button("Analisis Risiko Mahasiswa", use_container_width=True)

    if submit:
        # URUTAN HARUS SAMA DENGAN INDEKS DI GAMBAR (0 s/d 9)
        expected_columns = [
            'Course',                               # Index 0
            'Previous_qualification_grade',         # Index 1
            'Admission_grade',                      # Index 2
            'Tuition_fees_up_to_date',              # Index 3
            'Age_at_enrollment',                    # Index 4
            'Curricular_units_1st_sem_approved',    # Index 5
            'Curricular_units_1st_sem_grade',       # Index 6
            'Curricular_units_2nd_sem_evaluations', # Index 7
            'Curricular_units_2nd_sem_approved',    # Index 8
            'Curricular_units_2nd_sem_grade'        # Index 9
        ]

        input_data = {
            'Course': MAP_COURSE[course],
            'Previous_qualification_grade': prev_grade,
            'Admission_grade': adm_grade,
            'Tuition_fees_up_to_date': 1 if tuition == "Ya" else 0,
            'Age_at_enrollment': age,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_2nd_sem_evaluations': sem2_eval,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade
        }

        df_input = pd.DataFrame([input_data])[expected_columns]

        try:
            X_scaled = model_data['scaler'].transform(df_input)
            prediction = model_data['model'].predict(X_scaled)
            
            if prediction[0] == 0:
                st.error("### Hasil: Berisiko Tinggi DROPOUT")
            else:
                st.success("### Hasil: Berpotensi LULUS (GRADUATE)")
                st.balloons()
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")