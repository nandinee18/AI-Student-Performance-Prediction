import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Driven Student Performance Prediction")
st.write(
    "Predict a student's final academic score using "
    "academic, engagement, and lifestyle factors."
)


# Load dataset
data = pd.read_csv("student_data.csv")

features = [
    "Previous_Semester_GPA",
    "Past_Exam_Score",
    "Attendance_Rate",
    "Assignment_Submission_Rate",
    "Weekly_Study_Hours",
    "Sleep_Duration"
]

target = "Final_Score"

X = data[features]
y = data[target]


# Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Model evaluation
predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    previous_gpa = st.number_input(
        "Previous Semester GPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    exam_score = st.number_input(
        "Past Exam Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    attendance = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

with col2:
    assignment_rate = st.number_input(
        "Assignment Submission Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    study_hours = st.number_input(
        "Weekly Study Hours",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=0.5
    )

    sleep_duration = st.number_input(
        "Sleep Duration (hours/night)",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )


if st.button("🔮 Predict Performance"):

    student = pd.DataFrame({
        "Previous_Semester_GPA": [previous_gpa],
        "Past_Exam_Score": [exam_score],
        "Attendance_Rate": [attendance],
        "Assignment_Submission_Rate": [assignment_rate],
        "Weekly_Study_Hours": [study_hours],
        "Sleep_Duration": [sleep_duration]
    })

    prediction = model.predict(student)[0]

    prediction = max(
        0,
        min(100, prediction)
    )

    if prediction >= 75:
        category = "Excellent"
    elif prediction >= 60:
        category = "Good"
    elif prediction >= 50:
        category = "Average"
    else:
        category = "Needs Improvement"

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Final Score",
        f"{prediction:.2f}%"
    )

    if category == "Excellent":
        st.success(f"Performance Category: {category}")
    elif category == "Good":
        st.info(f"Performance Category: {category}")
    elif category == "Average":
        st.warning(f"Performance Category: {category}")
    else:
        st.error(f"Performance Category: {category}")


with st.expander("Model Information"):

    st.write(
        "The model uses Random Forest Regression "
        "trained on the student performance dataset."
    )

    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    st.write("Features used:")

    for feature in features:
        st.write(f"- {feature}")