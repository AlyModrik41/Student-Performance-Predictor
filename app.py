import streamlit as st
import joblib
import numpy as np

# ​​​ Load your pre-trained model and scaler
model = joblib.load("model.pkl")     # trained Ridge/Lasso/GBR
scaler = joblib.load("scaler.pkl")   # StandardScaler fit on training data
feature_names = joblib.load("features.pkl")  # order of features expected by the model

st.set_page_config(page_title="Student Performance Predictor")
st.title("🎓 Student Exam Score Predictor")
st.write("Enter your data below to predict your exam score:")

# ​​​ Input form
with st.form(key="input_form"):
    hours = st.number_input("Hours Studied", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 80, step=1)
    parental_inv = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_res = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    extr = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    prev = st.number_input("Previous Score", 0.0, 100.0, 70.0, step=1.0)
    internet = st.selectbox("Internet Access at Home", ["No", "Yes"])
    tutoring = st.number_input("Number of Tutoring Sessions", 0, 10, 0, step=1)
    phys_act = st.number_input("Physical Activity Hours/Week", 0.0, 20.0, 3.0, step=0.5)
    family_inc = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    peer_inf = st.selectbox("Peer Influence", ["Negative",'Neutral','Positive'])
    disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
    parent_ed = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance = st.selectbox("Distance from Home ",['Far','Moderate','Near'])
    gender = st.selectbox("Gender", ["Male", "Female"])
    submitted = st.form_submit_button("Predict Exam Score")

# ​​​ Process and predict
if submitted:
    # Prepare input data
    ord_map = {"Low": 0, "Medium": 1, "High": 2,
               "No": 0, "Yes": 1,
               "Public": 0, "Private": 1,
               "High School": 0, "College": 1, "Postgraduate": 2,
               "Male": 0, "Female": 1
              ,"Negative":0,'Neutral':1,'Positive':2,
              'Far':0,'Moderate':1,'Near':2}

    x = [
        hours, attendance,          # num
        ord_map[parental_inv],
        ord_map[access_res],
        ord_map[motivation],
        ord_map[internet],
        ord_map[family_inc],
        ord_map[teacher_quality],
        ord_map[school_type],
        ord_map[peer_inf],
        ord_map[disabilities],
        ord_map[parent_ed],
        sleep, prev, tutoring, phys_act, ord_map[distance],
        ord_map[extr]
    ]

    # scale numeric + ordinal as done in training
    x_arr = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x_arr)

    # get prediction
    pred = model.predict(x_scaled)[0]
    st.success(f"🎯 Predicted Exam Score: {pred:.1f} / 100")
