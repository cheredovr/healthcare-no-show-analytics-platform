"""
ЭТАП 1-3: EDA анализ
"""
import sys

sys.path.append('.')
from src.preprocessing import load_raw_data, clean_data, save_clean_data
from src.features import create_features
from src.eda import *


def run_eda():
    # Загрузка и очистка
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)

    # Feature Engineering
    df_feat = create_features(df_clean)
    save_clean_data(df_feat)

    # EDA графики
    plot_no_show_distribution(df_feat)
    plot_no_show_by_age(df_feat)
    plot_no_show_by_lead_time(df_feat)
    plot_correlation_heatmap(df_feat)

    # Сохранение summary таблиц
    summary = df_feat.describe()
    summary.to_csv('outputs/tables/01_eda/data_summary.csv')
    print("✅ EDA завершён")
