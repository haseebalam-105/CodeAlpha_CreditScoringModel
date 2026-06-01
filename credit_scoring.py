# =============================================================================
# CREDIT SCORING MODEL — CodeAlpha ML Internship Task 1
# Author: [Your Name]
# Description: Predict individual creditworthiness using classification models
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563EB', '#DC2626']

# =============================================================================
# STEP 1: DATA GENERATION (Synthetic Dataset)
# =============================================================================
# In a real project, replace this section with your actual dataset loading:
#   df = pd.read_csv('your_dataset.csv')
# Popular real datasets: German Credit Dataset (UCI), Lending Club, Kaggle Credit

def generate_dataset(n=2000, seed=42):
    """Generate a realistic synthetic credit scoring dataset."""
    np.random.seed(seed)

    age = np.random.randint(18, 70, n)
    income = np.random.normal(55000, 25000, n).clip(10000, 200000).astype(int)
    loan_amount = np.random.normal(15000, 10000, n).clip(1000, 80000).astype(int)
    credit_history_years = np.random.randint(0, 30, n)
    num_credit_accounts = np.random.randint(1, 15, n)
    num_late_payments = np.random.poisson(1.5, n)
    debt_to_income = np.round(np.random.beta(2, 5, n) * 0.9, 3)
    employment_status = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed', 'Retired'], n,
        p=[0.55, 0.2, 0.15, 0.1]
    )
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], n,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    num_inquiries = np.random.poisson(2, n)
    savings = np.random.exponential(10000, n).clip(0, 100000).astype(int)

    # Creditworthiness composite score
    score = (
        (income / 200000) * 30
        + (credit_history_years / 30) * 20
        + (1 - debt_to_income) * 20
        + (1 - num_late_payments / 20) * 15
        + (savings / 100000) * 10
        + (num_credit_accounts / 15) * 5
        - (num_inquiries / 20) * 5
        + np.where(employment_status == 'Employed', 5, 0)
        + np.where(education == 'PhD', 3, np.where(education == 'Master', 2, 0))
    )

    prob = 1 / (1 + np.exp(-0.15 * (score - 50)))
    target = (prob + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    return pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'credit_history_years': credit_history_years,
        'num_credit_accounts': num_credit_accounts,
        'num_late_payments': num_late_payments,
        'debt_to_income_ratio': debt_to_income,
        'employment_status': employment_status,
        'education_level': education,
        'num_credit_inquiries': num_inquiries,
        'savings_balance': savings,
        'creditworthy': target
    })


print("=" * 60)
print("  CREDIT SCORING MODEL — CodeAlpha ML Internship Task 1")
print("=" * 60)

df = generate_dataset()
df.to_csv('credit_data.csv', index=False)
print(f"\n[1] Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n[2] Exploratory Data Analysis")
print("-" * 40)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['creditworthy'].value_counts())
print(f"\n  Creditworthy rate: {df['creditworthy'].mean() * 100:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Scoring — EDA Overview', fontsize=15, fontweight='bold', y=1.02)

# 1. Class distribution
ax = axes[0, 0]
counts = df['creditworthy'].value_counts()
bars = ax.bar(['Not Creditworthy', 'Creditworthy'], counts, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(count), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Target Class Distribution', fontweight='bold')
ax.set_ylabel('Count')

# 2. Income distribution by class
ax = axes[0, 1]
for label, color in zip([0, 1], COLORS):
    ax.hist(df[df['creditworthy'] == label]['income'] / 1000,
            bins=30, alpha=0.65, color=color, label=['Not Creditworthy', 'Creditworthy'][label])
ax.set_title('Income Distribution by Class', fontweight='bold')
ax.set_xlabel('Income (thousands)')
ax.legend()

# 3. Debt-to-income by class
ax = axes[0, 2]
for label, color in zip([0, 1], COLORS):
    ax.hist(df[df['creditworthy'] == label]['debt_to_income_ratio'],
            bins=30, alpha=0.65, color=color, label=['Not Creditworthy', 'Creditworthy'][label])
ax.set_title('Debt-to-Income Ratio by Class', fontweight='bold')
ax.set_xlabel('Debt-to-Income Ratio')
ax.legend()

# 4. Employment status
ax = axes[1, 0]
emp_ct = df.groupby(['employment_status', 'creditworthy']).size().unstack()
emp_ct.plot(kind='bar', ax=ax, color=COLORS, edgecolor='white', linewidth=1)
ax.set_title('Employment Status vs Creditworthiness', fontweight='bold')
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.legend(['Not Creditworthy', 'Creditworthy'])

# 5. Credit history years
ax = axes[1, 1]
for label, color in zip([0, 1], COLORS):
    ax.hist(df[df['creditworthy'] == label]['credit_history_years'],
            bins=20, alpha=0.65, color=color, label=['Not Creditworthy', 'Creditworthy'][label])
ax.set_title('Credit History Years by Class', fontweight='bold')
ax.set_xlabel('Years')
ax.legend()

# 6. Correlation heatmap
ax = axes[1, 2]
num_cols = ['age', 'income', 'loan_amount', 'credit_history_years',
            'num_late_payments', 'debt_to_income_ratio', 'savings_balance', 'creditworthy']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={'shrink': 0.7}, linewidths=0.5)
ax.set_title('Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: eda_overview.png")

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
print("\n[3] Feature Engineering")
print("-" * 40)

def feature_engineer(data):
    df_fe = data.copy()

    # Derived features
    df_fe['loan_to_income_ratio']  = df_fe['loan_amount'] / df_fe['income']
    df_fe['savings_to_income_ratio'] = df_fe['savings_balance'] / df_fe['income']
    df_fe['payment_reliability']   = 1 / (1 + df_fe['num_late_payments'])
    df_fe['credit_utilization']    = df_fe['num_credit_inquiries'] / (df_fe['num_credit_accounts'] + 1)
    df_fe['financial_stability']   = (df_fe['income'] - df_fe['loan_amount']) / df_fe['income']

    # Age bucket
    df_fe['age_group'] = pd.cut(df_fe['age'],
        bins=[17, 25, 35, 50, 70],
        labels=['18-25', '26-35', '36-50', '51+']
    )

    # Encode categoricals
    le = LabelEncoder()
    for col in ['employment_status', 'education_level', 'age_group']:
        df_fe[col + '_enc'] = le.fit_transform(df_fe[col].astype(str))
        df_fe.drop(columns=[col], inplace=True)

    return df_fe

df_eng = feature_engineer(df)
features = [c for c in df_eng.columns if c != 'creditworthy']
X = df_eng[features]
y = df_eng['creditworthy']

print(f"  Original features : {df.shape[1] - 1}")
print(f"  After engineering : {len(features)}")
print(f"  New features added: loan_to_income_ratio, savings_to_income_ratio,")
print(f"                      payment_reliability, credit_utilization, financial_stability, age_group")

# =============================================================================
# STEP 4: TRAIN / TEST SPLIT & SCALING
# =============================================================================
print("\n[4] Train/Test Split & Scaling")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train size: {X_train.shape[0]} samples")
print(f"  Test size : {X_test.shape[0]} samples")

# =============================================================================
# STEP 5: MODEL TRAINING
# =============================================================================
print("\n[5] Training Models")
print("-" * 40)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    # Use scaled data for LR, unscaled for tree-based
    Xtr = X_train_sc if 'Logistic' in name else X_train
    Xte = X_test_sc  if 'Logistic' in name else X_test

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1]

    results[name] = {
        'model':    model,
        'X_test':   Xte,
        'y_pred':   y_pred,
        'y_prob':   y_prob,
        'cv_auc':   cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'test_auc': roc_auc_score(y_test, y_prob),
        'report':   classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"\n  {name}")
    print(f"    CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test AUC: {roc_auc_score(y_test, y_prob):.4f}")

# =============================================================================
# STEP 6: EVALUATION & VISUALIZATION
# =============================================================================
print("\n[6] Model Evaluation")
print("-" * 40)

# --- Metrics summary ---
print(f"\n{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-" * 58)
for name, res in results.items():
    r = res['report']['weighted avg']
    print(f"  {name:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1-score']:>8.4f} {res['test_auc']:>8.4f}")

best_name = max(results, key=lambda k: results[k]['test_auc'])
print(f"\n  Best model: {best_name} (AUC = {results[best_name]['test_auc']:.4f})")

# --- Plot 1: ROC curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

ax = axes[0]
palette = ['#2563EB', '#16A34A', '#DC2626', '#D97706']
for (name, res), color in zip(results.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{name} (AUC={res['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=9)

# --- Plot 2: CV AUC comparison ---
ax = axes[1]
names = list(results.keys())
aucs  = [results[n]['cv_auc'] for n in names]
stds  = [results[n]['cv_std'] for n in names]
bars  = ax.barh(names, aucs, xerr=stds, color=palette,
                error_kw={'ecolor': 'gray', 'capsize': 4}, edgecolor='white')
ax.set_xlabel('Cross-Validated AUC')
ax.set_title('5-Fold CV AUC Comparison', fontweight='bold')
ax.set_xlim(0.5, 1.0)
for bar, v in zip(bars, aucs):
    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: model_evaluation.png")

# --- Plot 2: Confusion matrix + Feature importance (best model) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=14, fontweight='bold')

best = results[best_name]
cm   = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Creditworthy', 'Creditworthy'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontweight='bold')

# Feature importance (tree-based models)
if hasattr(best['model'], 'feature_importances_'):
    importances = pd.Series(best['model'].feature_importances_, index=features)
    top15 = importances.nlargest(15).sort_values()
    top15.plot(kind='barh', ax=axes[1], color='#2563EB', edgecolor='white')
    axes[1].set_title('Top 15 Feature Importances', fontweight='bold')
    axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('best_model_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: best_model_analysis.png")

# --- Precision-Recall curve ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for (name, res), color in zip(results.items(), palette):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(rec, prec, lw=2, color=color, label=name)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: precision_recall.png")

# =============================================================================
# STEP 7: SAMPLE PREDICTION (Inference demo)
# =============================================================================
print("\n[7] Sample Prediction Demo")
print("-" * 40)

def predict_creditworthiness(applicant_dict, model, scaler_obj=None):
    """Predict creditworthiness for a new applicant."""
    df_app = pd.DataFrame([applicant_dict])
    df_app = feature_engineer(df_app)
    # Align columns (drop target if present)
    df_app = df_app[[c for c in features if c in df_app.columns]]
    for c in features:
        if c not in df_app.columns:
            df_app[c] = 0
    df_app = df_app[features]
    if scaler_obj:
        df_app = scaler_obj.transform(df_app)
    prob = model.predict_proba(df_app)[0][1]
    label = 'CREDITWORTHY ✓' if prob >= 0.5 else 'NOT CREDITWORTHY ✗'
    return prob, label

applicants = [
    {
        'name': 'High-Risk Applicant',
        'age': 24, 'income': 22000, 'loan_amount': 18000,
        'credit_history_years': 1, 'num_credit_accounts': 2,
        'num_late_payments': 5, 'debt_to_income_ratio': 0.72,
        'employment_status': 'Unemployed', 'education_level': 'High School',
        'num_credit_inquiries': 6, 'savings_balance': 500,
    },
    {
        'name': 'Low-Risk Applicant',
        'age': 45, 'income': 90000, 'loan_amount': 12000,
        'credit_history_years': 18, 'num_credit_accounts': 7,
        'num_late_payments': 0, 'debt_to_income_ratio': 0.18,
        'employment_status': 'Employed', 'education_level': 'Master',
        'num_credit_inquiries': 1, 'savings_balance': 40000,
    },
]

best_model = results[best_name]['model']
use_scaler  = scaler if 'Logistic' in best_name else None

for app in applicants:
    name_key = app.pop('name')
    prob, label = predict_creditworthiness(app, best_model, use_scaler)
    print(f"\n  {name_key}")
    print(f"    Creditworthy probability : {prob * 100:.1f}%")
    print(f"    Decision                 : {label}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Dataset          : 2,000 samples, {len(features)} features")
print(f"  Train/Test split : 80% / 20%  (stratified)")
print(f"  Best Model       : {best_name}")
print(f"  Best AUC         : {results[best_name]['test_auc']:.4f}")
print(f"  Outputs saved    : eda_overview.png, model_evaluation.png,")
print(f"                     best_model_analysis.png, precision_recall.png")
print("=" * 60)
print("\n  Project complete! Upload to GitHub as:")
print("  CodeAlpha_CreditScoringModel")
print("=" * 60)
