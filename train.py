import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score

# Load data
data = pd.read_csv("data/scoring.csv")
data["income_log"] = np.log1p(data["income"])

# scaling does not improve results
# numerical_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# numerical_columns = ["age", "income", "income_log"]

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_columns )
#     ], remainder='passthrough'
# )

# X_preprocessed = preprocessor.fit_transform(X)

X = data.drop(columns=["default"])
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss", random_state=42)
}

param_grids = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear"]
    },
    "DecisionTree": {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_model_name = ""
best_test_recall = 0

grids = {}
for model_name, model in models.items():
    grids[model_name] = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )
    
    grids[model_name].fit(X_train, y_train)
    
    best_params = grids[model_name].best_params_
    best_cv_score = grids[model_name].best_score_
    
    y_pred = grids[model_name].predict(X_test)
    # focus on recall for the positive class (default=1) cause we want to catch as many defaulters as possible + 90/10 class imbalance
    test_recall = recall_score(y_test, y_pred, pos_label=1)
    
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best CV Recall: {best_cv_score:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    if test_recall > best_test_recall:
        best_test_recall = test_recall
        best_model = grids[model_name].best_estimator_
        best_model_name = model_name

print(f"\nBest model: {best_model_name} | Test Recall = {best_test_recall:.4f}")

y_pred = best_model.predict(X_test)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)

print(f"Denied: {y_pred.mean() * 100:.0f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

joblib.dump(best_model, "models/best_model.pkl")

