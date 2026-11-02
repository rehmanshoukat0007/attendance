import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.joblib"
DATA_PATH = BASE_DIR / "archive_2.zip"

st.set_page_config(page_title="Student Performance System", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "random_forest_model.joblib is missing. Upload it beside app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv_from_zip(csv_name):
    if not DATA_PATH.exists():
        raise FileNotFoundError("archive_2.zip is missing. Upload it beside app.py.")
    with zipfile.ZipFile(DATA_PATH, "r") as archive:
        if csv_name not in archive.namelist():
            raise FileNotFoundError(f"{csv_name} was not found inside archive_2.zip.")
        with archive.open(csv_name) as file:
            return pd.read_csv(file)


def clean_percent(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


st.title("🎓 Student Performance Prediction System")
st.caption("Random Forest model | Exam score estimation and student attendance overview")

try:
    model = load_model()
    performance = load_csv_from_zip("performance.csv")
    attendance = load_csv_from_zip("attendance.csv")
    students = load_csv_from_zip("students.csv")
except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.info("Keep app.py, random_forest_model.joblib and archive_2.zip in the same GitHub repository folder.")
    st.stop()

performance["Homework_Completion_%"] = performance["Homework_Completion_%"].apply(clean_percent)
performance["Exam_Score"] = pd.to_numeric(performance["Exam_Score"], errors="coerce")
performance = performance.dropna(subset=["Subject", "Exam_Score"])
performance["Homework_Completion_%"] = performance["Homework_Completion_%"].fillna(
    performance["Homework_Completion_%"].median()
)

# LabelEncoder uses alphabetical ordering. The training script fitted it on
# the training split; the full dataset contains the same subject categories.
subjects = sorted(performance["Subject"].dropna().astype(str).unique().tolist())
subject_to_code = {subject: index for index, subject in enumerate(subjects)}

attendance["Attendance_Status_clean"] = (
    attendance["Attendance_Status"].astype(str).str.strip().str.upper()
)
attendance["Is_Present"] = attendance["Attendance_Status_clean"].str.contains(
    "PRES", regex=False
).astype(int)
attendance_summary = (
    attendance.groupby("Student_ID")["Is_Present"].mean().mul(100).rename("Attendance_Rate_%")
)

tab_predict, tab_dashboard, tab_data = st.tabs(
    ["📈 Predict Exam Score", "📊 Dashboard", "📁 Dataset Preview"]
)

with tab_predict:
    st.subheader("Estimate an exam score")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID (optional)", placeholder="e.g. S00001")
        subject = st.selectbox("Subject", subjects)
    with col2:
        homework = st.slider("Homework completion (%)", min_value=0, max_value=100, value=80)
        st.write("The model uses homework completion and subject as its input features.")

    if student_id.strip():
        student_row = students[students["Student_ID"].astype(str) == student_id.strip()]
        if not student_row.empty:
            row = student_row.iloc[0]
            st.success(f"Student: {row['Full_Name']} | Grade: {row['Grade_Level']}")
            rate = attendance_summary.get(student_id.strip())
            if rate is not None:
                st.metric("Attendance rate", f"{rate:.1f}%")
        else:
            st.warning("Student ID was not found. You can still make a prediction.")

    if st.button("Predict exam score", type="primary"):
        if subject not in subject_to_code:
            st.error("This subject is not available in the model's dataset.")
        else:
            input_df = pd.DataFrame(
                [[float(homework), subject_to_code[subject]]],
                columns=["Homework_Completion_%", "Subject_encoded"],
            )
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, min(100.0, prediction))
            st.metric("Predicted exam score", f"{prediction:.1f} / 100")
            st.caption("This is an estimate from the trained model, not a guaranteed exam result.")

with tab_dashboard:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{students['Student_ID'].nunique():,}")
    c2.metric("Performance records", f"{len(performance):,}")
    c3.metric("Attendance records", f"{len(attendance):,}")
    c4.metric("Average exam score", f"{performance['Exam_Score'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        st.write("**Average exam score by subject**")
        by_subject = performance.groupby("Subject")["Exam_Score"].mean().sort_values(ascending=False)
        st.bar_chart(by_subject)
    with right:
        st.write("**Top 10 attendance rates**")
        top_attendance = attendance_summary.sort_values(ascending=False).head(10).reset_index()
        top_attendance["Attendance_Rate_%"] = top_attendance["Attendance_Rate_%"].round(1)
        st.dataframe(top_attendance, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Performance data")
    st.dataframe(performance.head(100), use_container_width=True)
    st.caption("Showing the first 100 performance records.")
