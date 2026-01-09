"""
Exploratory Data Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_no_show_distribution(df, output_dir='outputs/figures/01_eda'):
    """Распределение no-show"""
    fig, ax = plt.subplots(figsize=(8, 5))
    no_show_counts = df['No-show'].value_counts()
    no_show_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Распределение No-Show', fontsize=14, weight='bold')
    ax.set_xlabel('No-Show (0=Пришёл, 1=Не пришёл)')
    ax.set_ylabel('Количество')
    ax.set_xticklabels(['Show', 'No-Show'], rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/no_show_distribution.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/no_show_distribution.png")


def plot_no_show_by_age(df, output_dir='outputs/figures/01_eda'):
    """No-show по возрасту"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column='Age', by='No-show', ax=ax)
    ax.set_title('Распределение возраста по No-Show')
    ax.set_xlabel('No-Show (0=Show, 1=No-Show)')
    ax.set_ylabel('Возраст')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/no_show_by_age.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/no_show_by_age.png")


def plot_no_show_by_lead_time(df, output_dir='outputs/figures/01_eda'):
    """No-show по lead_time_days"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df['lead_time_days'] <= 60].boxplot(column='lead_time_days', by='No-show', ax=ax)
    ax.set_title('Lead Time по No-Show (до 60 дней)')
    ax.set_xlabel('No-Show')
    ax.set_ylabel('Дней до приёма')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/no_show_by_lead_time.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/no_show_by_lead_time.png")


def plot_correlation_heatmap(df, output_dir='outputs/figures/01_eda'):
    """Тепловая карта корреляций"""
    numeric_cols = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism',
                    'Handicap', 'SMS_received', 'No-show', 'lead_time_days']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Матрица корреляций', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/correlation_heatmap.png")
