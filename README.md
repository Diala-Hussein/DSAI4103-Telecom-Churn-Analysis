# Telecom Customer Churn Prediction & Analytics

## Project Overview
This project implements an end-to-end **Business Analytics and Machine Learning pipeline** to analyze and predict customer churn in the telecom industry.

The objective is to identify customers likely to leave the service and provide actionable insights to support retention strategies.

---

## Business Problem
Telecom companies face significant revenue loss due to customer churn. Predicting churn allows businesses to:

- Improve customer retention
- Reduce financial losses
- Understand key drivers of churn behavior

---

## Dataset
The project uses the **Telecom Customer Churn by Maven Analytics** dataset from Kaggle.

### Dataset Files:
- `telecom_customer_churn.csv` → Main dataset (customer info, services, churn)
- `telecom_zipcode_population.csv` → Population data by zipcode
- `telecom_data_dictionary.csv` → Column descriptions

---

## Methodology

### 1.Data Preprocessing
- Removed irrelevant columns:
  - `Churn Reason`, `Churn Category`, `Offer`
- Filtered dataset:
  - Removed "Joined" customers
- Handled missing values:
  - Service features → filled with `"No"`
  - Numeric features → filled with median or `0`
- Created target variable:
  - `Churn = 1 (Churned), 0 (Stayed)`

---

### 2.Feature Engineering
- Dropped `Customer ID`
- Applied **One-Hot Encoding** to categorical variables
- Final dataset:
  - **6589 rows**
  - **1141 features**

---

### 3.Model Development
- Model used: **Random Forest Classifier**
- Data split:
  - 80% training
  - 20% testing

---

### 4.Model Performance

- **Accuracy:** 85.7%

#### Classification Results:
| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Stayed | 0.86 | 0.96 | 0.91 |
| Churned | 0.86 | 0.61 | 0.71 |

Insight:
- Model performs strongly overall
- Slightly weaker at detecting churn (class imbalance)

---

### 5.Model Explainability (SHAP)
- Used SHAP to interpret feature importance
- Identified key drivers of churn such as:
  - Tenure
  - Contract type
  - Monthly charges
  - Number of referrals

---

### 6.Bias Analysis
- Evaluated fairness across gender groups

Results:
- Similar churn rates across males and females
- No significant bias detected
- Model predictions are consistent across groups

---

## Dashboard (Power BI)
An interactive Power BI dashboard was developed to visualize insights:

### Features:
- Customer churn overview
- Revenue and customer insights
- Contract type analysis
- Internet service analysis
- Interactive filters (Gender, Contract, Internet Service)


---

## Technologies Used
- Python (Pandas, NumPy)
- Scikit-learn
- SHAP (Explainability)
- Matplotlib / Seaborn
- Power BI

---

## Conclusion
This project demonstrates a complete business analytics workflow:

✔ Data cleaning and preprocessing  
✔ Feature engineering  
✔ Predictive modeling  
✔ Model explainability  
✔ Bias analysis  
✔ Interactive dashboard  

The results provide valuable insights into customer churn and support data-driven decision-making.

---

## Author
**Diala Hussein**  
DSAI 4103 – Business Analytics
