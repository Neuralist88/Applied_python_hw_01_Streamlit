import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import requests
import streamlit as st


def temperature_analysis(data: pd.DataFrame, city: str) -> dict:
    """
    Функция для расчета описательных статистик на основе исорических данных о температуре
    """

    data = data.copy()
    # Преобразование даты в формат datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])  
    # Устанавливаем время в качестве индекса
    data.set_index('timestamp', inplace=True)  
    # Отфильтруем данные для заданного города
    data = data[data['city'] == city]
    # Вычисление скользящего среднего с окном в 30 дней для каждого сезона
    data['moving_average'] = data.groupby('season')['temperature'].transform(lambda x: x.rolling(window='30D', min_periods=1).mean())
    # Получение средних значений скользящего среднего для каждого сезона
    seasonal_mov_avg = data.groupby('season')['moving_average'].last()  # Получаем последнее значение для каждого сезона
    # Вычисление стандартного отклонения c окном 5 дней для каждого сезона
    data['moving_std_dev'] = data.groupby('season')['temperature'].transform(lambda x: x.rolling(window='5D').std())
    seasonal_mov_std = data.groupby('season')['moving_std_dev'].last()  # Получаем последнее значение для каждого сезона    
    # Получение "профиля сезона". 
    data['mean_temp'] = data.groupby(['season'])['temperature'].transform('mean')
    seasonal_mean_temp = data.groupby('season')['mean_temp'].last()
    data['std_temp'] = data.groupby(['season'])['temperature'].transform('std')
    seasonal_std_temp = data.groupby('season')['std_temp'].last()
    # Вычисление минимальной температуры для каждого сезона
    data['min_temp'] = data.groupby(['season'])['temperature'].transform('min')
    seasonal_min_temp = data.groupby('season')['min_temp'].last()
    # Вычисление максимальной температуры для каждого сезона
    data['max_temp'] = data.groupby(['season'])['temperature'].transform('max')
    seasonal_max_temp = data.groupby('season')['max_temp'].last()

    # определим функцию для определения является ли температура аномальной
    data.reset_index(inplace=True)
    anomalous_temp = []
    anomalous_temp_dates = []
    for i in range(len(data.index)):
        if data.loc[i, 'temperature'] < data.loc[i, 'mean_temp'] - 2 * data.loc[i, 'std_temp'] or \
           data.loc[i, 'temperature'] > data.loc[i, 'mean_temp'] + 2 * data.loc[i, 'std_temp']:
            anomalous_temp.append(data.loc[i, 'temperature']) 
            anomalous_temp_dates.append(data.loc[i, 'timestamp'])     
    
    # Определение тренда изменения температуры
    # Преобразование timestamp в числовой формат (количество дней с начальной даты)
    data['days_since_start'] = np.arange(len(data))
    X = data[['days_since_start']].values  # Признак (дни)
    y = data['temperature'].values         # Целевая переменная (температура)
    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)    
    # Предсказание тренда
    y_pred = model.predict(X)

    # Определение направления тренда (положительный или отрицательный)
    if model.coef_ < 0:
        trend_dir = 'negative'
    elif model.coef_ > 0:
        trend_dir = 'positive'
    else:
        trend_dir = 'constant'

    # Собираем все результаты воедино
    output_summary = {}
    seasons = ['autumn', 'spring', 'summer', 'winter']    
    for season, roll_avg, roll_std, mean, std, min, max in zip(seasons,
                                                          seasonal_mov_avg,
                                                          seasonal_mov_std,
                                                          seasonal_mean_temp,
                                                          seasonal_std_temp,
                                                          seasonal_min_temp,
                                                          seasonal_max_temp):
        
        output_summary[season+'_Скользящее среднее'] = roll_avg
        output_summary[season+'_Скользящее станд. отклонение'] = roll_std
        output_summary[season+'_Средняя температура'] = mean
        output_summary[season+'_Станд. отклонение'] = std
        output_summary[season+'_Мин. температура'] = min
        output_summary[season+'_Мaкс. температура'] = max
        output_summary['Направление тренда'] = trend_dir
        output_summary['Точки тренда'] = y_pred
        output_summary['Все температуры'] = y
        output_summary['Aномальные температуры'] = anomalous_temp        
        output_summary['Даты'] = data['timestamp'] 
        output_summary['Даты аномальных температур'] = anomalous_temp_dates        

    return output_summary


def print_results(output_summary: dict)-> pd.DataFrame:
    """
    Функция, возвращающая описательные статистики по сезонам в виде датафрейма
    """

    dict_for_print = {'winter': [], 'spring': [], 'summer': [], 'autumn': []}

    for key in output_summary.keys():
        if key.startswith('winter'):          
            dict_for_print['winter'].append(output_summary[key])
        elif key.startswith('spring'):            
            dict_for_print['spring'].append(output_summary[key])
        elif key.startswith('summer'):            
            dict_for_print['summer'].append(output_summary[key])
        elif key.startswith('autumn'):              
            dict_for_print['autumn'].append(output_summary[key])

    indexes = ['Скользящее среднее', 'Скользящее станд. отклонение', 'Средняя температура', 'Станд. отклонение', 'Мин. температура', 'Мaкс. температура' ]
    output_df = pd.DataFrame(dict_for_print , index=indexes)
    return output_df


# Функция для построения графика
def plot_results(temp_analysis_output: dict, show_anomalies: bool) -> plt.Figure:
    """
    Функция, строящая график изменения температуры на основании анализа температуры
    """
    y = temp_analysis_output['Все температуры']
    y_pred = temp_analysis_output['Точки тренда']
    dates = temp_analysis_output['Даты']

    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=(14, 6))
    # Построение графиков
    ax.plot(dates, y, label='Исходные данные', color='blue', alpha=0.5)
    ax.plot(dates, y_pred, label='Линейный тренд', color='red', linewidth=2)

    if show_anomalies:
        ax.scatter(
            temp_analysis_output['Даты аномальных температур'],
            temp_analysis_output['Aномальные температуры'],
            label='Аномальные температуры',
            color='orange',
            zorder=5
        )

    # Оформление графика
    ax.set_title('Долгосрочный тренд изменения температуры')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Температура')
    ax.legend()
    ax.grid()

    return fig


def get_temperature_by_api(city, key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    try:
        # Отправка запроса
        response = requests.get(url)
    
        # Проверка статуса ответа
        if response.status_code == 200:
            # Если запрос успешный
            weather_data = response.json()
            temperature = weather_data['main']['temp']            
            return (f"Текущая температура в {city}: {temperature}°C", temperature)
            # return (temperature, 'OK')            
            
        else:
            # Если произошла ошибка
            error_data = response.json()  # Получение JSON ошибки            
            return (f"Ошибка: {error_data}", None)  # Печать всей ошибки в формате JSON

    except requests.exceptions.RequestException as e:
        return (f"Ошибка соединения: {e}", None)


def check_curr_temp(temp_analysis_output: dict, temperature: float) -> None:
    """
    Функция для проверки вводимой температуры для заданного города на аномальность
    """
     # Оперделим текущий сезон
    month_season = {1: 'winter',
                    2: 'winter',
                    3: 'spring',
                    4: 'spring',
                    5: 'spring',
                    6: 'summer',
                    7: 'summer',
                    8: 'summer',
                    9: 'autumn',
                    10: 'autumn',
                    11: 'autumn',
                    12: 'winter'}

    curr_season = month_season[datetime.datetime.now().month] 
    # Получаем исторические данные    
    historical_mean = temp_analysis_output[curr_season +'_Средняя температура']
    historical_std = temp_analysis_output[curr_season +'_Станд. отклонение']
    minimal_boundary = historical_mean - 2 * historical_std
    maximal_boundary = historical_mean + 2 * historical_std
    if minimal_boundary <= temperature <= maximal_boundary:
        return f'Текущая температура {temperature:.2f}°C попадает в диапазон [{minimal_boundary:.2f}°C, {maximal_boundary:.2f}°C] и не является аномальной'
    else:
        return f'Текущая температура {temperature:.2f}°C не попадает в диапазон [{minimal_boundary:.2f}°C, {maximal_boundary:.2f}°C] и является аномальной'