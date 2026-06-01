Step 1 — Data Generation: Synthetic 2,000-sample dataset with 11 realistic features (income, debt-to-income ratio, payment history, savings, employment, education, etc.). Swap this with a real CSV if you have one.
Step 2 — EDA: Visualizes class distribution, income/debt distributions by class, employment breakdown, credit history, and a full correlation heatmap.
Step 3 — Feature Engineering: Adds 6 derived features: loan_to_income_ratio, savings_to_income_ratio, payment_reliability, credit_utilization, financial_stability, and age_group buckets.
Step 4 — Preprocessing: Stratified 80/20 train-test split + StandardScaler for Logistic Regression.
Step 5 — Models trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting — all with 5-fold cross-validation.
Step 6 — Evaluation: ROC-AUC, Precision, Recall, F1-Score, confusion matrix, feature importances, PR curves.
