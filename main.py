import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Загрузка данных
df_red_wine = pd.read_csv('winequality-red.csv', sep=';')
# Очистим названия столбцов от лишних символов
df_red_wine.columns = df_red_wine.columns.str.strip().str.replace('"', '')

# Проверка на наличие пропущенных значений
print("Пропущенные значения в данных о красном вине:")
print(df_red_wine.isnull().sum())

# Преобразуем метки качества вина в бинарные классы
df_red_wine['good_wine'] = df_red_wine['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Проверка первых строк данных
print(df_red_wine.head())


# Функция для нахождения выбросов
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Удаление выбросов по столбцу 'quality'
df_red_wine_no_outliers = remove_outliers(df_red_wine, 'quality')

print("Количество строк после удаления выбросов (красное вино):", len(df_red_wine_no_outliers))

# График распределения по качеству
sns.histplot(df_red_wine_no_outliers['quality'], kde=False)
plt.title('Распределение качества красного вина')
plt.xlabel('Качество')
plt.ylabel('Частота')
plt.show()

# График распределения бинарных меток
sns.countplot(x='good_wine', data=df_red_wine_no_outliers)
plt.title('Баланс бинарных классов (хорошее и плохое вино)')
plt.xlabel('Класс вина (0 = Плохое, 1 = Хорошее)')
plt.ylabel('Частота')
plt.show()


# Медианы для всех признаков в наборе данных о красном вине
medians_red_wine = df_red_wine_no_outliers.median()
print("Медианы по каждому признаку (красное вино):")
print(medians_red_wine)

# Ящик с усами для столбца 'quality'
sns.boxplot(x='quality', data=df_red_wine_no_outliers)
plt.title('Ящик с усами по показателю качества (красное вино)')
plt.xlabel('Качество')
plt.show()


"""Графики распределения:

В цикле мы строим гистограммы с наложенными кривыми плотности (kde) для каждого признака.
Используется функция sns.histplot из библиотеки Seaborn для построения распределений.
Матрица корреляции:

Вычисляем корреляцию с помощью функции df_red_wine.corr().
Для визуализации используем тепловую карту (heatmap) с аннотацией коэффициентов корреляции, чтобы было видно взаимосвязь между признаками.
"""
# Построим гистограммы распределения для каждого признака, включая 'quality'
features = df_red_wine.columns[:-1]  # Исключаем только 'good_wine'

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df_red_wine[feature], kde=True)
    plt.title(f'Распределение: {feature}')
plt.tight_layout()
plt.show()


# Построим матрицу корреляции
plt.figure(figsize=(12, 8))
correlation_matrix = df_red_wine.corr()

# Используем тепловую карту (heatmap) для визуализации корреляции
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Матрица корреляции признаков (красное вино)')
plt.show()

