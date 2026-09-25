import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('phones.csv')

df = df.drop(columns=['Отметка времени'])

#Названия столбцов

df.columns = [
    'stability', 'photo_freq', 'blog', 'pay_extra', 'status_tech',
    'vpn', 'age', 'budget', 'custom_ui', 'ringtone', 'resale',
    'system_tune', 'change_freq', 'os_support', 'sphere', 'gender',
    'instagram', 'charge_times', 'save_or_spend', 'headset',
    'apples', 'target'
]

print("Исходный размер данных:", df.shape)
print("\nРаспределение классов:")
print(df['target'].value_counts())


 #Очистка бюджета (если указано значение НЕ в тысячах)

def clean_budget(x):
    try:
        val = float(str(x).replace(' ', '').replace(',', '.'))
        if val < 1000: 
            val *= 1000
        return val
    except:
        return np.nan

df['budget'] = df['budget'].apply(clean_budget)


#Преобразование в числа и удаление выбросов
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['change_freq'] = pd.to_numeric(df['change_freq'], errors='coerce')
df['custom_ui'] = pd.to_numeric(df['custom_ui'], errors='coerce')
df['charge_times'] = pd.to_numeric(df['charge_times'], errors='coerce')
df['apples'] = pd.to_numeric(df['apples'], errors='coerce')


df = df[(df['age'] >= 15) & (df['age'] <= 50)]
df = df[df['budget'] >= 10000]

print("\nПосле очистки:", df.shape)

#Кодирование категориальных признаков 

cat_cols = ['stability', 'photo_freq', 'blog', 'status_tech', 'vpn',
            'ringtone', 'resale', 'system_tune', 'os_support',
            'gender', 'instagram', 'save_or_spend', 'headset']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

#Сфера деятельности
le_sphere = LabelEncoder()
df['sphere'] = le_sphere.fit_transform(df['sphere'].astype(str).str.strip().str.lower())

#Мультивыбор "За что готовы переплатить"
pay_options = ['Статус', 'Камера', 'Экосистема', 'Надежность', 'Мультивыбор']
for opt in pay_options:
    df[f'pay_{opt}'] = df['pay_extra'].astype(str).str.contains(opt, case=False).astype(int)

df = df.drop(columns=['pay_extra'])

#Целевая переменная
le_target = LabelEncoder()
df['target'] = le_target.fit_transform(df['target'])

print("\nКлассы:", list(le_target.classes_))

X = df.drop(columns=['target'])
y = df['target']

print("Количество признаков:", X.shape[1])

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка:  {X_test.shape[0]} примеров")

#Обучение и оценка k-NN
print("\n--- Результаты при разных k ---")
for k in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k = {k}  →  Accuracy = {acc:.3f}")


best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\n===== Итоговый результат (k = {best_k}) =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))