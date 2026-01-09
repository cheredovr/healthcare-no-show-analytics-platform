# 🏥 Healthcare No-Show Analytics Platform

> **Аналитическая платформа для прогнозирования неявки пациентов на медицинские приёмы**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io/)

---

## 📋 Содержание

- [О проекте](#-о-проекте)
- [Описание задачи](#-описание-задачи)
- [Структура проекта](#-структура-проекта)
- [Установка и запуск](#-установка-и-запуск)
- [Этапы пайплайна](#-этапы-пайплайна)
- [Результаты](#-результаты)
- [Дашборд](#-дашборд)
- [Ключевые инсайты](#-ключевые-инсайты)
- [Технологии](#-технологии)

---

## 🎯 О проекте

Данный проект представляет собой **end-to-end аналитическую платформу**, которая решает бизнес-проблему высокой доли пропущенных медицинских приёмов (no-show). Платформа включает:

- ✅ Полный ETL-пайплайн обработки данных
- ✅ Exploratory Data Analysis (EDA) с визуализациями
- ✅ Обучение и сравнение 4 ML-моделей
- ✅ Интерактивный web-дашборд на Streamlit
- ✅ Калькулятор риска no-show для отдельных пациентов

**Цель:** Снизить долю неявок на 5-10% за счёт приоритизации напоминаний пациентам с высоким риском.

---

## 📊 Описание задачи

### Проблема
20-30% пациентов не приходят на назначенные приёмы, что приводит к:
- Потере рабочего времени врачей
- Снижению доступности медпомощи для других пациентов
- Финансовым потерям клиники

### Решение
Построение ML-модели, которая предсказывает вероятность no-show на основе данных о пациенте и записи, что позволяет:
- Отправлять дополнительные напоминания группе риска
- Оптимизировать расписание с учётом прогнозируемых неявок
- Приоритизировать звонки пациентам с высокой вероятностью no-show

---

## 📁 Структура проекта

```
healthcare-no-show-analytics-platform/
│
├── main.py                          # 🎯 Главная точка входа (запуск всего пайплайна)
├── app.py                           # 📊 Streamlit дашборд
├── requirements.txt                 # 📦 Зависимости
├── README.md                        # 📖 Документация (этот файл)
│
├── data/                            # 📊 Данные
│   ├── raw/                         # Исходные данные (CSV с Kaggle)
│   │   └── KaggleV2-May-2016.csv
│   └── processed/                   # Обработанные данные
│       ├── appointments_clean.csv
│       ├── X_train_final.csv
│       ├── X_test_final.csv
│       ├── y_train_final.csv
│       ├── y_test_final.csv
│       └── feature_names_final.txt
│
├── src/                             # 💻 Исходный код (модули)
│   ├── __init__.py
│   ├── preprocessing.py             # Загрузка и очистка данных
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── features.py                  # Feature Engineering
│   └── modeling.py                  # Обучение и оценка моделей
│
├── pipeline/                        # 🔄 Этапы пайплайна
│   ├── __init__.py
│   ├── eda_analysis.py              # ЭТАП 1-3: EDA
│   ├── feature_selection.py         # ЭТАП 4: Анализ признаков
│   └── model_training.py            # ЭТАП 5-7: Обучение моделей
│
└── outputs/                         # 📤 Результаты
    ├── figures/                     # 📊 Графики
    │   ├── 01_eda/                  # EDA визуализации
    │   ├── 02_feature_analysis/     # Анализ признаков
    │   └── 03_model_results/        # Результаты моделей
    │       ├── roc_curves_comparison.png
    │       ├── confusion_matrices_comparison.png
    │       ├── metrics_comparison.png
    │       └── feature_importance_*.png
    │
    ├── tables/                      # 📋 Таблицы
    │   ├── 01_eda/
    │   ├── 02_feature_analysis/
    │   └── 03_model_results/
    │       ├── model_comparison_all.csv
    │       └── feature_importance_*.csv
    │
    └── models/                      # 🤖 Обученные модели
        ├── LogisticRegression_baseline.pkl
        ├── LogisticRegression_final.pkl
        ├── RandomForest_final.pkl
        ├── GradientBoosting_final.pkl   # ⭐ Лучшая модель
        └── XGBoost_final.pkl
```

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/your-username/healthcare-no-show-analytics-platform.git
cd healthcare-no-show-analytics-platform
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

Основные библиотеки:
- `pandas`, `numpy` — обработка данных
- `scikit-learn` — ML-модели
- `xgboost` — Gradient Boosting
- `imbalanced-learn` — балансировка классов (SMOTE)
- `matplotlib`, `seaborn`, `plotly` — визуализация
- `streamlit` — web-дашборд

### 3. Загрузка данных
Скачайте датасет с Kaggle: 🔗 [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

Положите файл в:
```
data/raw/KaggleV2-May-2016.csv
```

### 4. Запуск полного пайплайна
```bash
python main.py
```

Что выполнится:
- ✅ Загрузка и очистка данных (110k → 72k записей)
- ✅ Feature Engineering (создание 5 новых признаков)
- ✅ EDA с визуализациями
- ✅ Обучение 4 моделей с SMOTE-балансировкой
- ✅ Сохранение результатов в `outputs/`

**Время выполнения:** ~2-3 минуты

### 5. Запуск дашборда
```bash
streamlit run app.py
```
Откроется браузер на `http://localhost:8501` 🎉

---

## 🔄 Этапы пайплайна

### ЭТАП 1-3: EDA (Exploratory Data Analysis)
**Цель:** Понять данные, найти паттерны и выбросы

**Входные данные:** `data/raw/KaggleV2-May-2016.csv`

**Действия:**
- Загрузка данных (110,527 записей)
- Очистка:
  - Удаление дубликатов по AppointmentID
  - Фильтрация аномалий возраста (< 0 или > 110)
  - Удаление записей, где дата записи > даты приёма
  - Преобразование дат в datetime
  - Создание целевой переменной: No-show (Yes/No → 1/0)

**Результат:** ~72,000 чистых записей

**Визуализации:**
- Распределение no-show (28.5% неявок)
- No-show по возрасту, по lead_time, по SMS
- Матрица корреляций

📊 **Графики:** `outputs/figures/01_eda/`

### ЭТАП 4: Feature Engineering
**Цель:** Создать новые признаки для улучшения модели

**Созданные признаки:**
- `lead_time_days` — дней между записью и приёмом (ключевой признак!)
- `appointment_dow` — день недели приёма (0=Пн, 6=Вс)
- `scheduled_dow` — день недели записи
- `scheduled_hour` — час записи
- `appointment_month` — месяц приёма
- `age_group` — возрастные группы (child, young_adult, adult, senior, elderly)

**Итоговые признаки для модели (13):**
- Age, Gender, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMS_received
- lead_time_days, appointment_dow, scheduled_dow, scheduled_hour, appointment_month

### ЭТАП 5-7: Обучение и оценка моделей
**Цель:** Построить лучшую модель для прогноза no-show

**Модели:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting ⭐
- XGBoost

**Балансировка классов:**
- Исходное распределение: 71.5% Show / 28.5% No-Show
- SMOTE: увеличение класса No-Show до 50%
- Train/Test split: 80% / 20% (stratified)

**Метрики оценки:**
- ROC-AUC (основная метрика)
- Accuracy
- Precision/Recall для класса No-Show

---

## 📈 Результаты

### Сравнение моделей

| Модель | ROC-AUC | Accuracy | Примечание |
|--------|---------|----------|------------|
| **Gradient Boosting** ⭐ | 0.567 | 58.8% | Лучшая модель |
| XGBoost | 0.565 | 58.5% | Близко к GB |
| Random Forest | 0.562 | 58.2% | Стабильная |
| Logistic Regression | 0.537 | 55.2% | Baseline |

### Выводы:
- **Gradient Boosting** показал лучший результат (ROC-AUC 0.567)
- Прирост к baseline: +3% ROC-AUC, +3.6% Accuracy
- Все древовидные модели (RF, GB, XGBoost) показали схожие результаты

### Практическая ценность:
- Модель лучше случайного угадывания (AUC > 0.5)
- Позволяет приоритизировать 40-45% пациентов с высоким риском
- Ожидаемое снижение no-show: 5-10% при внедрении системы

---

## 🎨 Дашборд

Интерактивный web-дашборд на Streamlit включает:

### 📊 Вкладка "Аналитика"
- KPI-карточки: всего записей, неявки, доля no-show, средний lead time
- Фильтры: возраст, пол, SMS, lead_time
- Графики:
  - Pie chart: распределение no-show
  - Bar chart: влияние SMS на неявки
  - Line chart: no-show vs lead_time
  - Histogram: распределение возраста

### 🎯 Вкладка "Калькулятор риска"
- Ввод параметров пациента (возраст, пол, SMS, lead_time и др.)
- Прогноз вероятности no-show в реальном времени
- Gauge chart для визуализации риска
- Автоматические рекомендации (SMS/звонок)

### 📈 Вкладка "Сравнение моделей"
- Таблица метрик всех 4 моделей
- ROC-кривые на одном графике
- Confusion matrices
- Feature importance для лучшей модели

### ℹ️ Вкладка "О проекте"
- Цель и описание
- Этапы пайплайна
- Ключевые инсайты
- Практические рекомендации

---

## 💡 Ключевые инсайты

### 1. Lead Time — фактор #1
Чем дальше запись от даты приёма, тем выше риск no-show:
- Lead time < 7 дней: ~22% no-show
- Lead time 14-30 дней: ~32% no-show
- Lead time > 30 дней: ~38% no-show

📌 **Рекомендация:** Дополнительные напоминания для записей с lead_time > 14 дней

### 2. SMS-напоминания снижают неявки
Пациенты с SMS приходят чаще:
- Без SMS: ~30% no-show
- С SMS: ~25% no-show

📌 **Рекомендация:** Отправлять SMS всем пациентам за 24-48 часов до приёма

### 3. Молодые пациенты — группа риска
Возрастная группа 18-35 лет пропускает приёмы чаще:
- 18-35 лет: ~32% no-show
- 35-50 лет: ~27% no-show
- 50+ лет: ~22% no-show

📌 **Рекомендация:** Приоритетные звонки молодым пациентам с lead_time > 7 дней

### 4. День недели имеет значение
Пятница и понедельник — дни с повышенной неявкой

📌 **Рекомендация:** Резервировать буферные слоты в эти дни

---

## 🛠️ Технологии

**Data Processing:**
- Python 3.8+
- Pandas, NumPy

**Machine Learning:**
- scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- XGBoost
- imbalanced-learn (SMOTE)

**Visualization:**
- Matplotlib, Seaborn
- Plotly

**Web Dashboard:**
- Streamlit

**Development:**
- Jupyter Notebook (для экспериментов)
- Git/GitHub

---

## 📚 Дополнительно

### Возможные улучшения

**Feature Engineering:**
- История пропусков пациента (требует PatientId join)
- Расстояние до клиники (требует геоданные)
- Погода в день приёма (API)

**Модели:**
- Hyperparameter tuning (GridSearchCV)
- Ансамбли моделей (Stacking)
- Deep Learning (LSTM для временных паттернов)

**Deployment:**
- Контейнеризация (Docker)
- CI/CD pipeline
- Production API (FastAPI)

---

## 👨‍💻 Автор

**[Ваше имя]**
- 📧 Email: your.email@example.com
- 🔗 LinkedIn: linkedin.com/in/yourprofile
- 🐙 GitHub: github.com/yourusername

---

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

---

## 🙏 Благодарности

- Kaggle за публичный датасет [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- scikit-learn и XGBoost команды за отличные библиотеки
- Streamlit за простой фреймворк для web-приложений

---

<p align="center">
<b>Сделано с ❤️ для улучшения здравоохранения</b>
</p>
