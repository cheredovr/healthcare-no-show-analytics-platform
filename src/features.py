"""
Feature Engineering: создание признаков для моделирования
"""
import pandas as pd
import numpy as np


def create_features(df):
    """Создание новых признаков"""
    df = df.copy()

    # Lead time (дней между записью и приёмом)
    df['lead_time_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    # День недели приёма (0=Monday, 6=Sunday)
    df['appointment_dow'] = df['AppointmentDay'].dt.dayofweek

    # День недели записи
    df['scheduled_dow'] = df['ScheduledDay'].dt.dayofweek

    # Час записи (если есть время)
    df['scheduled_hour'] = df['ScheduledDay'].dt.hour

    # Месяц приёма
    df['appointment_month'] = df['AppointmentDay'].dt.month

    # Возрастные группы
    df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 120],
                             labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])

    print(f"✅ Создано {5} новых признаков")
    return df


def prepare_model_data(df):
    """Подготовка данных для моделирования"""
    # Выбираем признаки
    feature_cols = ['Age', 'Gender', 'Scholarship', 'Hypertension', 'Diabetes',
                    'Alcoholism', 'Handicap', 'SMS_received', 'lead_time_days',
                    'appointment_dow', 'scheduled_dow', 'scheduled_hour', 'appointment_month']

    # Кодирование Gender
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    X = df[feature_cols].copy()
    y = df['No-show'].copy()

    return X, y, feature_cols
