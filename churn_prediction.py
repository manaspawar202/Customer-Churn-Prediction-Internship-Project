import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

print("--- 1. Loading Data ---")
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:\n", df.head())
except FileNotFoundError:
    print("Error: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory as the script.")
    exit()

print("\n--- 2. Preprocessing Data ---")
df.drop('customerID', axis=1, inplace=True)
print("Dropped 'customerID' column.")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total_charges = df['TotalCharges'].isnull().sum()
print(f"Missing values in 'TotalCharges' after coercion: {missing_total_charges}")

if missing_total_charges > 0:
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    print(f"Imputed missing 'TotalCharges' with median value: {median_total_charges:.2f}")

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("Converted 'Churn' column to binary (1=Yes, 0=No).")

numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"\nIdentified Numerical Features: {numerical_features}")
print(f"Identified Categorical Features: {categorical_features}")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("\nPreprocessor created.")

print("\n--- 3. Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
print(f"Churn distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Churn distribution in testing set:\n{y_test.value_counts(normalize=True)}")

print("\n--- 4. Training Models ---")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

trained_pipelines = {}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline
    print(f"{name} trained.")

print("\n--- 5. Evaluating Models ---")
for name, pipeline in trained_pipelines.items():
    print(f"\n--- Evaluating {name} ---")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": roc_auc
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Churn=Yes): {precision:.4f}")
    print(f"Recall (Churn=Yes): {recall:.4f}")
    print(f"F1-Score (Churn=Yes): {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n--- Model Performance Summary ---")
results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = "Gradient Boosting"
print(f"\n--- Confusion Matrix for {best_model_name} ---")
best_pipeline = trained_pipelines[best_model_name]
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - {best_model_name}')
confusion_path = 'confusion_matrix_gb.png'
plt.tight_layout()
plt.savefig(confusion_path)
plt.close()
print(f"Saved confusion matrix plot to {confusion_path}")

print(f"\n--- 6. Feature Importance ({best_model_name}) ---")
preprocessor_step = best_pipeline.named_steps['preprocessor']
model_step = best_pipeline.named_steps['classifier']

if hasattr(model_step, 'feature_importances_'):
    ohe_transformer = preprocessor_step.named_transformers_['cat']
    ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_features)

    all_feature_names = numerical_features + list(ohe_feature_names)
    importances = model_step.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    top_n = 15
    print(f"Top {top_n} Features:\n", feature_importance_df.head(top_n))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - {best_model_name}')
    plt.tight_layout()
    importance_path = 'feature_importance_gb.png'
    plt.savefig(importance_path)
    plt.close()
    print(f"Saved feature importance plot to {importance_path}")
else:
    print(f"{best_model_name} does not directly provide feature importances through '.feature_importances_'.")

print("\n--- Script Finished ---")
