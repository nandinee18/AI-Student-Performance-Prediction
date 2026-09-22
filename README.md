# AI-Driven Student Performance Prediction System

An AI-based machine learning application that predicts a student's expected final academic score using academic, engagement, and lifestyle factors.

# CareerTrack

A career management dashboard built with Streamlit.

# Live Demo

View Live Demo → https://ai-student-performance-predi-yq2ygj93ixrbuex7uuzsfp.streamlit.app/

## Features

- Predicts expected final student score
- Classifies predicted performance into:
  - Excellent
  - Good
  - Average
  - Needs Improvement
- Interactive Streamlit interface
- Uses multiple student performance parameters
- Displays model evaluation metrics

## Parameters Used

The model uses six parameters:

- Previous Semester GPA
- Past Exam Score
- Attendance Rate
- Assignment Submission Rate
- Weekly Study Hours
- Sleep Duration

## Machine Learning

The project uses **Random Forest Regression** to predict the student's final score.

The dataset contains **500 synthetic student records** created for this project.

### Model Evaluation

- **Mean Absolute Error (MAE): 2.43**
- **R² Score: 0.91**

MAE represents the average difference between the predicted and actual scores.

R² indicates how well the model explains variation in the target scores.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Project Workflow

Student Information
↓
Data Preprocessing
↓
Random Forest Regression
↓
Predicted Final Score
↓
Performance Category

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
