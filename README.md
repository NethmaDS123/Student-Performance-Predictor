# Student-Performance-Predictor

This project uses a machine learning approach to predict student performance based on various features such as demographics, academic history, and personal attributes. By leveraging a neural network built with TensorFlow/Keras, the model predicts the final grade (G3) of students in a standardized format.

# Features
Data Preprocessing:

Encodes categorical variables using LabelEncoder.
Scales numerical features using StandardScaler.
Generates interaction features through polynomial feature expansion.

Neural Network Architecture:

Input layer tailored to the preprocessed data.
Three hidden layers with ReLU activation, dropout regularization, and L2 weight regularization to prevent overfitting.
Output layer predicts final grades with linear activation for regression.

Performance Metrics:

R² Score: Measures the model's goodness of fit.
Mean Absolute Percentage Error (MAPE): Evaluates the prediction error.
Tolerance-based accuracy: Percentage of predictions within ±10% and ±20% tolerance of actual values.
Visualizations:

Scatter plot comparing actual and predicted grades.
Training and validation loss curves across epochs.

Data
The dataset is sourced from the UCI Machine Learning Repository (Student Performance dataset) and contains academic and personal data of students. The dataset includes features like:

Demographics: Age, gender, address type.
Academic: Grades in previous exams (G1, G2), absences.
Personal: Parental education, support received, extracurricular activities, internet access, etc.
Key Steps in the Project
Data Preprocessing:

Encoded categorical data.
Scaled numerical data for optimal neural network performance.
Expanded features using polynomial feature generation.
Model Training:

Used a neural network with multiple layers and regularization techniques.
Optimized using the Adam optimizer.
Incorporated early stopping to prevent overfitting.

Evaluation:

Compared model predictions to actual values using metrics and tolerance-based accuracy.
Visualized the performance through plots and predictions.

Results
R² Score: Indicates how well the model explains the variability of the target variable.
MAPE: Provides an understanding of the percentage error in predictions.
Tolerance Accuracy: Demonstrates the model's precision within 10% and 20% ranges of actual grades.

Visualizations
Scatter Plot: Shows predicted vs. actual grades, along with a line representing perfect prediction.
Loss Curves: Demonstrate the convergence of training and validation loss during model training.

How to Run
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/student-performance-predictor.git
Install required libraries:
Copy code
pip install pandas numpy scikit-learn tensorflow matplotlib requests
Execute the Jupyter notebook or Python script.

# Acknowledgments
Dataset from the UCI Machine Learning Repository.
Developed using Python, TensorFlow/Keras, and Scikit-learn.
