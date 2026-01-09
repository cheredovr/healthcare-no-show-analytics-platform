"""
Загрузка и очистка данных
"""
import pandas as pd
import numpy as np


def load_raw_data(filepath='data/raw/KaggleV2-May-2016.csv'):
    """Загрузка сырых данных"""
    df = pd.read_csv(filepath)
    print(f"✅ Загружено {len(df)} записей, {df.shape[1]} колонок")
    return df


def clean_data(df):
    """Очистка данных"""
    df = df.copy()

    # Переименование колонок для единообразия
    df.columns = ['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
                  'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship',
                  'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap',
                  'SMS_received', 'No-show']

    # Преобразование дат
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Целевая переменная: No-show Yes->1, No->0
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

    # Удаление строк с отрицательным возрастом или аномалиями
    df = df[df['Age'] >= 0]
    df = df[df['Age'] <= 110]

    # Удаление дубликатов по AppointmentID
    df = df.drop_duplicates(subset='AppointmentID')

    # Удаление записей, где ScheduledDay > AppointmentDay (невозможная ситуация)
    df = df[df['ScheduledDay'] <= df['AppointmentDay']]

    print(f"✅ После очистки: {len(df)} записей")
    return df


def save_clean_data(df, filepath='data/processed/appointments_clean.csv'):
    """Сохранение очищенных данных"""
    df.to_csv(filepath, index=False)
    print(f"✅ Очищенные данные сохранены: {filepath}")
