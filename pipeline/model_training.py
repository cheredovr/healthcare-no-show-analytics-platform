"""
ЭТАП 5-7: Подготовка, обучение и оценка моделей (улучшенная версия)
"""
import sys
sys.path.append('.')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from src.features import prepare_model_data
from src.modeling import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    evaluate_model,
    save_model,
    apply_smote
)

def plot_comparison_charts(results_list, output_dir='outputs/figures/03_model_results'):
    """Графики сравнения моделей"""

    # 1. ROC Curves для всех моделей
    plt.figure(figsize=(10, 8))

    for result in results_list:
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Сравнение моделей', fontsize=14, weight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves_comparison.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/roc_curves_comparison.png")

    # 2. Confusion Matrices
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

    if n_models == 1:
        axes = [axes]

    for idx, result in enumerate(results_list):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   ax=axes[idx], cbar=False)
        axes[idx].set_title(result['model_name'], fontsize=12, weight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_comparison.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/confusion_matrices_comparison.png")

    # 3. Метрики (bar chart)
    metrics_df = pd.DataFrame([
        {'Model': r['model_name'], 'Accuracy': r['accuracy'], 'ROC-AUC': r['roc_auc']}
        for r in results_list
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].bar(metrics_df['Model'], metrics_df['Accuracy'], color='steelblue')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy по моделям', fontsize=13, weight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    # ROC-AUC
    axes[1].bar(metrics_df['Model'], metrics_df['ROC-AUC'], color='coral')
    axes[1].set_ylabel('ROC-AUC', fontsize=12)
    axes[1].set_title('ROC-AUC по моделям', fontsize=13, weight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/metrics_comparison.png")

def run_model_training():
    """Запуск обучения нескольких моделей с SMOTE"""

    df = pd.read_csv('data/processed/appointments_clean.csv')
    X, y, feature_names = prepare_model_data(df)

    print(f"\n📊 Исходное распределение классов:")
    print(y.value_counts())
    print(f"   Доля No-Show: {y.mean()*100:.1f}%")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Сохранение splits
    X_train.to_csv('data/processed/X_train_final.csv', index=False)
    X_test.to_csv('data/processed/X_test_final.csv', index=False)
    y_train.to_csv('data/processed/y_train_final.csv', index=False)
    y_test.to_csv('data/processed/y_test_final.csv', index=False)

    with open('data/processed/feature_names_final.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    # Применяем SMOTE для балансировки
    print("\n⚖️ Применение SMOTE для балансировки классов...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Список моделей для обучения
    models_to_train = {
        'LogisticRegression': train_logistic_regression,
        'RandomForest': train_random_forest,
        'GradientBoosting': train_gradient_boosting,
        'XGBoost': train_xgboost
    }

    results_list = []

    # Обучение и оценка всех моделей
    for model_name, train_func in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"🚀 Обучение модели: {model_name}")
        print(f"{'='*60}")

        # Обучаем на сбалансированных данных
        model = train_func(X_train_balanced, y_train_balanced)

        # Оцениваем на тестовых (несбалансированных) данных
        result = evaluate_model(model, X_test, y_test, feature_names, model_name)
        results_list.append(result)

        # Сохраняем модель
        model_path = f'outputs/models/{model_name}_final.pkl'
        save_model(model, model_path)

        # Сохраняем важность признаков
        if result['feature_importance'] is not None:
            importance_path = f'outputs/tables/03_model_results/feature_importance_{model_name}.csv'
            result['feature_importance'].to_csv(importance_path, index=False)

            # График важности для этой модели
            top_features = result['feature_importance'].head(10)
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'], color='teal')
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top-10 важных признаков ({model_name})', fontsize=14, weight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'outputs/figures/03_model_results/feature_importance_{model_name}.png', dpi=150)
            plt.close()

    # Сравнительные графики
    print(f"\n{'='*60}")
    print("📊 Построение сравнительных графиков...")
    print(f"{'='*60}")
    plot_comparison_charts(results_list)

    # Сводная таблица метрик
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'ROC-AUC': r['roc_auc']
        }
        for r in results_list
    ]).sort_values('ROC-AUC', ascending=False)

    comparison_df.to_csv('outputs/tables/03_model_results/model_comparison_all.csv', index=False)

    print(f"\n{'='*60}")
    print("📋 ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    # Лучшая модель
    best_model_name = comparison_df.iloc[0]['Model']
    best_roc_auc = comparison_df.iloc[0]['ROC-AUC']

    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")

    print("\n✅ Все модели обучены и оценены")
