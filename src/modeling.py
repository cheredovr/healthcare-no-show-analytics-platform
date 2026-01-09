"""
Обучение и оценка моделей
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import numpy as np

def train_logistic_regression(X_train, y_train):
    """Обучение baseline Logistic Regression"""
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("✅ Модель Logistic Regression обучена")
    return model

def train_random_forest(X_train, y_train):
    """Обучение Random Forest"""
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Модель Random Forest обучена")
    return model

def train_gradient_boosting(X_train, y_train):
    """Обучение Gradient Boosting"""
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Модель Gradient Boosting обучена")
    return model

def train_xgboost(X_train, y_train):
    """Обучение XGBoost"""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("✅ Модель XGBoost обучена")
    return model

def evaluate_model(model, X_test, y_test, feature_names, model_name='Model'):
    """Оценка модели"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Метрики {model_name}:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC-AUC:  {roc_auc:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))

    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        # Для Random Forest, Gradient Boosting, XGBoost
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # Для Logistic Regression
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        importance_df = None

    return {
        'model_name': model_name,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': importance_df,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }

def save_model(model, filepath):
    """Сохранение модели"""
    joblib.dump(model, filepath)
    print(f"✅ Модель сохранена: {filepath}")

def apply_smote(X_train, y_train):
    """Балансировка классов через SMOTE"""
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE применён: {len(X_train)} → {len(X_resampled)} записей")
    print(f"   Распределение классов: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled
