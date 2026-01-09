"""
ЭТАП 4: Анализ и выбор признаков
"""
import sys

sys.path.append('.')
import pandas as pd


def run_feature_selection():
    df = pd.read_csv('data/processed/appointments_clean.csv')

    # Корреляции с целевой переменной
    numeric_cols = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism',
                    'Handicap', 'SMS_received', 'lead_time_days', 'appointment_dow', 'scheduled_dow']
    corr_with_target = df[numeric_cols + ['No-show']].corr()['No-show'].drop('No-show').sort_values(ascending=False)

    corr_with_target.to_csv('outputs/tables/02_feature_analysis/feature_correlation_analysis.csv')
    print("✅ Feature selection завершён")
