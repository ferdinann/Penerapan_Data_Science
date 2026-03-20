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

# --- 2. FUNGSI LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    model_path = 'Model/student_performance_model.joblib'
    scaler_path = 'Model/student_scaler.joblib'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return {
            'model': joblib.load(model_path),
            'scaler': joblib.load(scaler_path)
        }
    return None

model_data = load_model()

# --- 3. MAPPING DATA ---
MAP_COURSE = {
    "Biofuel Production Technologies": 1, "Animation and Multimedia Design": 2,
    "Social Service": 3, "Agronomy": 4, "Communication Design": 5,
    "Veterinary Nursing": 6, "Informatics Engineering": 7, "Equiniculture": 8,
    "Management": 9, "Social Service (evening attendance)": 10,
    "Tourism": 11, "Nursing": 12, "Oral Hygiene": 13, "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15, "Basic Education": 16, "Management (evening attendance)": 17
}

MAP_FATHER_JOB = {
    "Tenaga Ahli/Teknis": 1, "Pekerja Kerah Biru": 3, 
    "Pekerja Kerah Putih": 5, "Petani/Nelayan": 10, "Lainnya": 0
}

# --- 4. UI HEADER ---
st.title("🎓 Student Dropout Prediction")
st.markdown("### Jaya Jaya Institut Monitoring System")
st.info("Prediksi ini menggunakan fitur utama yang paling berpengaruh terhadap keberhasilan mahasiswa.")
st.divider()

if model_data is None:
    st.error("File model atau scaler tidak ditemukan di folder 'Model/'. Pastikan path sudah benar.")
else:
    # --- 5. INPUT FORM ---
    with st.form("prediction_form"):
        st.subheader("Data Input Mahasiswa")
        col1, col2 = st.columns(2)
        
        with col1:
            course = st.selectbox("Program Studi", list(MAP_COURSE.keys()))
            prev_grade = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 120.0)
            father_job = st.selectbox("Pekerjaan Ayah", list(MAP_FATHER_JOB.keys()))
            adm_grade = st.number_input("Nilai Admission (Masuk)", 0.0, 200.0, 120.0)
            age = st.slider("Usia Saat Pendaftaran", 17, 60, 20)
            tuition_fees = st.radio("Uang Kuliah Lunas (Up to date)?", ["Ya", "Tidak"])

        with col2:
            sem1_approved = st.number_input("Unit Semester 1 Disetujui", 0, 30, 5)
            sem1_grade = st.number_input("IP/Nilai Semester 1", 0.0, 20.0, 12.0)
            sem2_eval = st.number_input("Jumlah Evaluasi Semester 2", 0, 50, 10)
            sem2_approved = st.number_input("Unit Semester 2 Disetujui", 0, 30, 5)
            sem2_grade = st.number_input("IP/Nilai Semester 2", 0.0, 20.0, 12.0)

        submit = st.form_submit_button("Analisis Risiko Mahasiswa", use_container_width=True)

    # --- 6. PROSES PREDIKSI ---
    if submit:
        expected_columns = [
            'Course',
            'Previous_qualification_grade',
            'Fathers_occupation',
            'Admission_grade',
            'Age_at_enrollment',
            'Tuition_fees_up_to_date', 
            'Curricular_units_1st_sem_approved',
            'Curricular_units_1st_sem_grade',
            'Curricular_units_2nd_sem_evaluations',
            'Curricular_units_2nd_sem_approved',
            'Curricular_units_2nd_sem_grade'
        ]

        input_data = {
            'Course': MAP_COURSE[course],
            'Previous_qualification_grade': prev_grade,
            'Fathers_occupation': MAP_FATHER_JOB[father_job],
            'Admission_grade': adm_grade,
            'Age_at_enrollment': age,
            'Tuition_fees_up_to_date': 1 if tuition_fees == "Ya" else 0,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_2nd_sem_evaluations': sem2_eval,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade
        }

        df_input = pd.DataFrame([input_data])
        
        # Memaksa urutan kolom agar sesuai dengan expected_columns
        df_input = df_input[expected_columns]

        try:
            # Scaling & Prediksi
            X_scaled = model_data['scaler'].transform(df_input)
            prediction = model_data['model'].predict(X_scaled)
            
            st.divider()
            
            # Mapping Hasil Prediksi (Sesuaikan dengan label encoding di notebook)
            if prediction[0] == 0:
                st.error("### Hasil: Berisiko Tinggi DROPOUT")
                st.info("**Rekomendasi:** Berikan bimbingan konseling dan bantuan finansial/akademik segera.")
            else:
                st.success("### Hasil: Berpotensi LULUS (GRADUATE)")
                st.balloons()
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
            st.write("Cek urutan kolom ini di Notebook kamu:", df_input.columns.tolist())