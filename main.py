import os
from pipeline.eda_analysis import run_eda
from pipeline.feature_selection import run_feature_selection
from pipeline.model_training import run_model_training


def main():
    print("=" * 60)
    print("АНАЛИТИЧЕСКАЯ ПЛАТФОРМА: ПРОГНОЗ NO-SHOW ПАЦИЕНТОВ")
    print("=" * 60)

    # Создаём папки для результатов
    os.makedirs("outputs/figures/01_eda", exist_ok=True)
    os.makedirs("outputs/figures/02_feature_analysis", exist_ok=True)
    os.makedirs("outputs/figures/03_model_results", exist_ok=True)
    os.makedirs("outputs/tables/01_eda", exist_ok=True)
    os.makedirs("outputs/tables/02_feature_analysis", exist_ok=True)
    os.makedirs("outputs/tables/03_model_results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ЭТАП 1-3: EDA
    print("\n[ЭТАП 1-3] Запуск EDA анализа...")
    run_eda()

    # ЭТАП 4: Feature Selection
    print("\n[ЭТАП 4] Анализ и выбор признаков...")
    run_feature_selection()

    # ЭТАП 5-7: Обучение и оценка моделей
    print("\n[ЭТАП 5-7] Подготовка, обучение и оценка моделей...")
    run_model_training()

    print("\n" + "=" * 60)
    print("✅ ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ!")
    print("Результаты сохранены в папке outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()