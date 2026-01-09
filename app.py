"""
Интерактивный дашборд для анализа no-show пациентов
Запуск: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

# Конфигурация страницы
st.set_page_config(
    page_title="No-Show Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных и модели
@st.cache_data
def load_data():
    """Загрузка обработанных данных"""
    df = pd.read_csv('data/processed/appointments_clean.csv')
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    return df

@st.cache_resource
def load_model():
    """Загрузка лучшей обученной модели (Gradient Boosting)"""
    return joblib.load('outputs/models/GradientBoosting_final.pkl')

@st.cache_data
def load_model_comparison():
    """Загрузка сравнения моделей"""
    return pd.read_csv('outputs/tables/03_model_results/model_comparison_all.csv')

# Загружаем данные и модель
df = load_data()
model = load_model()
model_comparison = load_model_comparison()

# ========== SIDEBAR: ФИЛЬТРЫ ==========
st.sidebar.header("🔍 Фильтры данных")

# Фильтр по возрасту
age_range = st.sidebar.slider(
    "Возраст пациента",
    int(df['Age'].min()),
    int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max()))
)

# Фильтр по полу
gender_options = ['Все'] + list(df['Gender'].unique())
gender_filter = st.sidebar.selectbox("Пол", gender_options)

# Фильтр по SMS
sms_options = ['Все', 'С SMS', 'Без SMS']
sms_filter = st.sidebar.selectbox("SMS-напоминание", sms_options)

# Фильтр по lead_time
lead_time_range = st.sidebar.slider(
    "Дней до приёма (lead time)",
    0,
    int(df['lead_time_days'].max()),
    (0, 30)
)

# Применение фильтров
df_filtered = df[
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1]) &
    (df['lead_time_days'] >= lead_time_range[0]) &
    (df['lead_time_days'] <= lead_time_range[1])
]

if gender_filter != 'Все':
    df_filtered = df_filtered[df_filtered['Gender'] == gender_filter]

if sms_filter == 'С SMS':
    df_filtered = df_filtered[df_filtered['SMS_received'] == 1]
elif sms_filter == 'Без SMS':
    df_filtered = df_filtered[df_filtered['SMS_received'] == 0]

# ========== ГЛАВНАЯ СТРАНИЦА ==========
st.title("🏥 Аналитическая платформа прогноза неявки пациентов")
st.markdown("---")

# Табы для разделов
tab1, tab2, tab3, tab4 = st.tabs(["📊 Аналитика", "🎯 Калькулятор риска", "📈 Сравнение моделей", "ℹ️ О проекте"])

# ========== TAB 1: АНАЛИТИКА ==========
with tab1:
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)

    total_appointments = len(df_filtered)
    no_show_count = df_filtered['No-show'].sum()
    no_show_rate = (no_show_count / total_appointments * 100) if total_appointments > 0 else 0
    avg_lead_time = df_filtered['lead_time_days'].mean()

    with col1:
        st.metric("Всего записей", f"{total_appointments:,}")
    with col2:
        st.metric("Неявки (No-Show)", f"{no_show_count:,}")
    with col3:
        st.metric("Доля неявок", f"{no_show_rate:.1f}%")
    with col4:
        st.metric("Ср. lead time", f"{avg_lead_time:.1f} дней")

    st.markdown("---")

    # Графики
    col_left, col_right = st.columns(2)

    with col_left:
        # Pie chart: распределение no-show
        no_show_dist = df_filtered['No-show'].value_counts()
        fig_pie = px.pie(
            values=no_show_dist.values,
            names=['Пришёл (Show)', 'Не пришёл (No-Show)'],
            title="Распределение No-Show",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Bar chart: no-show по SMS
        sms_noshow = df_filtered.groupby('SMS_received')['No-show'].mean() * 100
        fig_sms = px.bar(
            x=['Без SMS', 'С SMS'],
            y=sms_noshow.values,
            title="Доля неявок: SMS vs Без SMS",
            labels={'x': 'SMS-напоминание', 'y': 'Доля неявок (%)'},
            color=sms_noshow.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_sms, use_container_width=True)

    # График: no-show по lead_time
    st.subheader("📅 Неявки по lead time (дней до приёма)")
    lead_time_grouped = df_filtered[df_filtered['lead_time_days'] <= 60].groupby('lead_time_days')['No-show'].mean() * 100
    fig_lead = px.line(
        x=lead_time_grouped.index,
        y=lead_time_grouped.values,
        title="Доля неявок в зависимости от lead time (до 60 дней)",
        labels={'x': 'Дней до приёма', 'y': 'Доля неявок (%)'},
        markers=True
    )
    st.plotly_chart(fig_lead, use_container_width=True)

    # График: распределение возраста
    st.subheader("👤 Распределение возраста пациентов")
    fig_age = px.histogram(
        df_filtered,
        x='Age',
        color='No-show',
        nbins=30,
        title="Распределение возраста (Show vs No-Show)",
        labels={'No-show': 'Статус'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        barmode='overlay',
        opacity=0.7
    )
    st.plotly_chart(fig_age, use_container_width=True)

# ========== TAB 2: КАЛЬКУЛЯТОР РИСКА ==========
with tab2:
    st.header("🎯 Калькулятор риска неявки")
    st.markdown("Введите параметры пациента для прогноза вероятности no-show")

    st.info("🔬 **Используется модель:** Gradient Boosting (лучшая по ROC-AUC = 0.567)")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_age = st.number_input("Возраст", min_value=0, max_value=120, value=35)
        input_gender = st.selectbox("Пол", ['F', 'M'])
        input_scholarship = st.selectbox("Соц.программа (Scholarship)", [0, 1])
        input_hypertension = st.selectbox("Гипертония", [0, 1])

    with col2:
        input_diabetes = st.selectbox("Диабет", [0, 1])
        input_alcoholism = st.selectbox("Алкоголизм", [0, 1])
        input_handicap = st.number_input("Инвалидность (0-4)", min_value=0, max_value=4, value=0)
        input_sms = st.selectbox("SMS-напоминание", [0, 1])

    with col3:
        input_lead_time = st.number_input("Дней до приёма (lead time)", min_value=0, max_value=180, value=7)
        input_appt_dow = st.selectbox("День недели приёма (0=Пн, 6=Вс)", list(range(7)))
        input_sched_dow = st.selectbox("День недели записи (0=Пн, 6=Вс)", list(range(7)))
        input_sched_hour = st.number_input("Час записи (0-23)", min_value=0, max_value=23, value=10)

    input_appt_month = st.selectbox("Месяц приёма (1-12)", list(range(1, 13)))

    if st.button("🔮 Рассчитать риск", type="primary"):
        # Подготовка входных данных
        input_gender_encoded = 0 if input_gender == 'F' else 1

        input_data = pd.DataFrame({
            'Age': [input_age],
            'Gender': [input_gender_encoded],
            'Scholarship': [input_scholarship],
            'Hypertension': [input_hypertension],
            'Diabetes': [input_diabetes],
            'Alcoholism': [input_alcoholism],
            'Handicap': [input_handicap],
            'SMS_received': [input_sms],
            'lead_time_days': [input_lead_time],
            'appointment_dow': [input_appt_dow],
            'scheduled_dow': [input_sched_dow],
            'scheduled_hour': [input_sched_hour],
            'appointment_month': [input_appt_month]
        })

        # Прогноз
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")

        # Результат
        if prediction == 1:
            st.error(f"⚠️ **Высокий риск неявки!** Вероятность no-show: **{probability:.1f}%**")
            st.markdown("**Рекомендации:**")
            st.markdown("- ✉️ Отправить дополнительное SMS-напоминание за 1-2 дня")
            st.markdown("- 📞 Позвонить пациенту для подтверждения")
            st.markdown("- 📋 Добавить в список приоритетного контроля")
        else:
            st.success(f"✅ **Низкий риск неявки.** Вероятность no-show: **{probability:.1f}%**")
            st.markdown("**Рекомендации:**")
            st.markdown("- 📧 Стандартное SMS-напоминание за 24 часа")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Вероятность No-Show (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ========== TAB 3: СРАВНЕНИЕ МОДЕЛЕЙ ==========
with tab3:
    st.header("📈 Сравнение моделей")

    # Таблица сравнения
    st.subheader("📊 Метрики всех моделей")
    st.dataframe(
        model_comparison.style.highlight_max(subset=['ROC-AUC', 'Accuracy'], color='lightgreen'),
        use_container_width=True
    )

    # Графики сравнения
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart метрик
        st.image('outputs/figures/03_model_results/metrics_comparison.png',
                 caption='Сравнение метрик (Accuracy и ROC-AUC)',
                 use_container_width=True)

    with col2:
        # ROC curves
        st.image('outputs/figures/03_model_results/roc_curves_comparison.png',
                 caption='ROC-кривые всех моделей',
                 use_container_width=True)

    # Confusion matrices
    st.subheader("🔍 Матрицы ошибок (Confusion Matrices)")
    st.image('outputs/figures/03_model_results/confusion_matrices_comparison.png',
             caption='Confusion Matrices для всех моделей',
             use_container_width=True)

    # Важность признаков для лучшей модели
    st.subheader("⭐ Важность признаков (Gradient Boosting)")
    try:
        st.image('outputs/figures/03_model_results/feature_importance_GradientBoosting.png',
                 caption='Top-10 важных признаков для лучшей модели',
                 use_container_width=True)

        # Таблица важности
        importance_df = pd.read_csv('outputs/tables/03_model_results/feature_importance_GradientBoosting.csv')
        st.dataframe(importance_df.head(15), use_container_width=True)
    except:
        st.warning("График важности признаков не найден")

# ========== TAB 4: О ПРОЕКТЕ ==========
with tab4:
    st.header("ℹ️ О проекте")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Цель проекта")
        st.markdown("""
        Создание аналитической платформы для прогнозирования неявки пациентов на медицинские приёмы 
        с целью оптимизации загрузки клиники и снижения потерь времени врачей.
        """)

        st.subheader("📊 Данные")
        st.markdown("""
        - **Источник:** Kaggle - Medical Appointment No Shows
        - **Размер:** 110,527 записей → 71,955 (после очистки)
        - **Признаков:** 13 (возраст, пол, SMS, болезни, lead_time и др.)
        - **Целевая переменная:** No-show (1 = неявка, 0 = пришёл)
        - **Дисбаланс классов:** 71.5% Show / 28.5% No-Show
        """)

    with col2:
        st.subheader("🤖 Модели")
        st.markdown("""
        Обучено **4 модели** с балансировкой классов (SMOTE):
        1. Logistic Regression (baseline)
        2. Random Forest
        3. **Gradient Boosting** ⭐ (лучшая)
        4. XGBoost
        
        **Лучший результат:**
        - ROC-AUC: **0.567**
        - Accuracy: **58.8%**
        """)

        st.subheader("📈 Этапы пайплайна")
        st.markdown("""
        1. ✅ Загрузка и очистка данных
        2. ✅ Feature Engineering (lead_time, день недели и др.)
        3. ✅ EDA (графики, корреляции)
        4. ✅ Обучение 4 моделей
        5. ✅ Сравнение и выбор лучшей
        6. ✅ Интерактивный дашборд (Streamlit)
        """)

    st.markdown("---")

    st.subheader("💡 Ключевые инсайты")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📅 Lead Time", "Фактор #1", delta="Чем дальше запись, тем выше риск")
    with col2:
        st.metric("📱 SMS-напоминание", "Снижает no-show", delta="-5% неявок")
    with col3:
        st.metric("👴 Возраст", "Молодые пропускают чаще", delta="18-35 лет — группа риска")

    st.markdown("---")

    st.subheader("🎓 Практическая ценность")
    st.success("""
    **Рекомендации для клиники:**
    - 📞 Приоритетные звонки пациентам с вероятностью no-show > 50%
    - 📧 Дополнительные SMS-напоминания для записей с lead_time > 14 дней
    - 📊 Мониторинг no-show rate по районам и корректировка расписания
    - 🎯 Ожидаемое снижение no-show на 5-10% при внедрении системы
    """)

    st.markdown("---")
    st.caption("© 2026 Healthcare No-Show Analytics Platform | Developed with ❤️ using Python, Streamlit, scikit-learn, XGBoost")
