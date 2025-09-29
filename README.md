# Heart Disease Prediction

A machine learning project that predicts **heart disease risk** based on patient data.  
This project uses **Logistic Regression** and **Decision Tree** classifiers to predict whether a patient is at risk of heart disease. The model with **higher accuracy** is automatically chosen as the decision maker.

---

## Features

- Predicts heart disease risk for patients in the test dataset.
- Automatically selects the **best model** based on accuracy.
- Computes **probabilities** and binary predictions (at risk / not at risk).
- Visualizes:
  - Feature importance for both Logistic Regression and Decision Tree
  - Confusion matrices
  - Number of patients at risk vs not at risk

---

## Dataset

- Uses the **Heart Disease dataset** (`heart.csv`).
- Features include various medical attributes such as age, cholesterol, blood pressure, etc.
- Target: `0` (No heart disease) or `1` (Heart disease present).

---

## How It Works

1. **Data Loading:** Reads `heart.csv` and splits into features (`X`) and target (`y`).
2. **Train-Test Split:** 80% training, 20% testing.
3. **Logistic Regression:**
   - Scales features using `StandardScaler`.
   - Trains on scaled training data.
   - Computes accuracy, confusion matrix, and classification report.
   - Visualizes feature importance and confusion matrix heatmap.
4. **Decision Tree:**
   - Trains on raw training data.
   - Computes accuracy, confusion matrix, and classification report.
   - Visualizes feature importance and confusion matrix heatmap.
5. **Decision Maker:**
   - Compares accuracy of Logistic Regression vs Decision Tree.
   - Chooses the higher accuracy model as the decision maker.
   - Computes predicted probabilities and final binary predictions.
6. **Risk Summary:**
   - Counts patients predicted **at risk** and **not at risk**.
   - Visualizes the results using a bar chart.

---

## Usage

1. Clone or download this repository.
2. Open the notebook in **Google Colab** or Jupyter Notebook.
3. Make sure `heart.csv` is in the correct path (`/content/sample_data/heart.csv` or update the path accordingly).
4. Run the notebook to see:
   - Accuracy and evaluation metrics for both models
   - Feature importance plots
   - Confusion matrices
   - Predicted number of patients at risk vs not at risk
   - Probabilities and predictions for individual patients

---

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`, `confusion_matrix`, `classification_report`)

---

## Output

- **Feature Importance Plots:** Identify which features contribute most to prediction.  
- **Confusion Matrices:** Shows how well each model performs.  
- **Patients Risk Bar Chart:** Visual summary of predicted heart disease risk.  

---

## Notes

- The notebook automatically picks the **best-performing model** based on accuracy.  
- Logistic Regression uses **scaled data**, while Decision Tree uses **raw values**.  
- Predictions and probabilities are generated for the **test dataset**.  

---

## Demo

Example bar chart for predicted patients at risk:

![Patients at Risk]<img width="671" height="493" alt="image" src="https://github.com/user-attachments/assets/e3bd114b-7c77-4aef-aa3f-911d02128f31" />


