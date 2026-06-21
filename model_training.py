import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, classification_report



def train_risk_model(training_matrix: pd.DataFrame, scale_weight: float):
    """
    Trains a LightGBM classification model using Time-Series Cross Validation.
    Evaluates strictly on Precision-Recall AUC to handle extreme class imbalance.
    """
    print("] Preparing data for modeling...")

    # 1. Ensure absolute chronological order to prevent data leakage
    df = training_matrix.sort_values(['target_time', 'location_uid']).reset_index(drop=True)

    # 2. Define Features (X) and Target (y)
    features = ['recency_minutes', 'countrywide_active', 'neighbor_active_count']
    X = df[features]
    y = df['target']

    # 3. Initialize Time-Series Split (e.g., 5 chronological folds)
    tscv = TimeSeriesSplit(n_splits=5)

    # 4. Initialize LightGBM Classifier with our calculated weight
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )

    pr_auc_scores = []
    fold = 1

    print("\n] Initiating Walk-Forward Cross-Validation...")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='aucpr',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Predict Probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate PR-AUC
        pr_auc = average_precision_score(y_test, y_pred_proba)
        pr_auc_scores.append(pr_auc)

        print(f"  -> Fold {fold} PR-AUC: {pr_auc:.4f}")
        fold += 1

    print(f"\n] Average PR-AUC across all folds: {np.mean(pr_auc_scores):.4f}")

    # Retrain on the entire dataset for production deployment
    print("] Retraining final production model on 100% of data...")
    model.fit(X, y)

    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance.to_string(index=False))

    return model

# To execute:
# Provide the scale weight calculated from the previous step
# production_model = train_risk_model(training_matrix, scale_weight=13.65)