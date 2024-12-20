import streamlit as st
import pandas as pd
import json
import functions


st.title('Анализ температурных данных и мониторинг текущей температуры')

st.header('Загрузка данных')

# Добавляем загрузку данных
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

# Чекбокс для отображения превью данных
show_preview= st.checkbox("Показать превью данных", value=False)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if show_preview == True:
        st.dataframe(data)
else:
    st.write("Пожалуйста, загрузите CSV-файл.")

# Добавляем возможность выбрать город
city = st.selectbox(
    "Выберите город из выпадающего списка?",
    ('New York',
     'London', 
     'Paris', 
     'Tokyo', 
     'Moscow', 
     'Sydney', 
     'Berlin', 
     'Beijing', 
     'Rio de Janeiro', 
     'Dubai', 'Los Angeles', 
     'Singapore', 'Mumbai', 
     'Cairo', 
     'Mexico City'),
)

st.write("Вы выбрали:", city)

# Инициализация состояния кнопок
if "button_1_clicked" not in st.session_state:
    st.session_state["button_1_clicked"] = False
if "button_2_clicked" not in st.session_state:
    st.session_state["button_2_clicked"] = False
if "button_3_clicked" not in st.session_state:
    st.session_state["button_3_clicked"] = False

# Функция для обработки нажатия кнопки
def toggle_button(button_key):
    st.session_state[button_key] = not st.session_state[button_key]

# Toggle button для запуска вычислений статистики температуры на основании исторических данных для выбранного города
st.button("Рассчитать статистику по историческим данным",
    key = 'button_1',
    on_click=toggle_button,
    args=("button_1_clicked",)  
)

if st.session_state["button_1_clicked"]:
    # Проводим анализ температуры на основании исторических данных
    temperature_results = functions.temperature_analysis(data, city)
    # Выводим описательные статистики в виде датафрейма
    output = functions.print_results(temperature_results)
    st.dataframe(output)


# Кнопка с сохранением состояния
st.button("Построить график температуры по историческим данным",
    key = 'button_2',
    on_click=toggle_button,
    args=("button_2_clicked",)  
)

# Чекбокс для отображения аномальных точек
show_anomalies = st.checkbox("Показать аномальные температуры", value=False)

# Toggle button  для отрисовки графика изменения температуры в течение времени наблюдений для выбранного города
if st.session_state["button_2_clicked"]:
    # Строим график изменения температуры    
    fig = functions.plot_results(temperature_results, show_anomalies)
    st.pyplot(fig)    


# Добавляем форму для ввода API ключа

# Форма для ввода ключа
with st.form(key="key_form"):
    key_entered = st.text_input("Введите API ключ:", type="password")  # Скрытое поле для ввода ключа
    submit_button = st.form_submit_button("Ввод")

# Подтверждаем ввод ключа (правильный он или нет будем проверять при обращении к API)
if submit_button:
    st.success("Вы ввели ключ!")   

# Toggle button для получения текущей температуры для выбранного города по API
st.button("Получить текущую температуру",
    key = 'button_3',
    on_click=toggle_button,
    args=("button_3_clicked",)  
)

if st.session_state["button_3_clicked"]:
    response = functions.get_temperature_by_api(city, key_entered)
    st.write(response[0])    
    # Чекбокс для проверки температуры на аномальность
    is_anomalous = st.checkbox("Проверить температуру на аномальнось", value=False)
    if is_anomalous:
        temperature = response[1]
        st.write(functions.check_curr_temp(temperature_results, temperature))