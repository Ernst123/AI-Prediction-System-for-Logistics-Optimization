import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
train_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv')

# 2. Очистка и подготовка данных
# Удаляем ненужные или неподдерживаемые столбцы
columns_to_drop = [
    'PROJECTNUMBER', 'invoiceDate', 'ItemDescription', 
    'MasterItemNo', 'UOM', 'ExtendedQuantity', 
    'PriceUOM', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE'
]
train_df = train_df.drop(columns=columns_to_drop, errors='ignore')

# Обработка числовых признаков
for col in ['invoiceTotal', 'UnitPrice', 'ExtendedPrice', 'QtyShipped']:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

# Обработка категориальных признаков
categorical_cols = ['PROJECT_CITY', 'STATE', 'PROJECT_COUNTRY', 'CORE_MARKET', 'PROJECT_TYPE']
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        joblib.dump(le, f'le_{col}.pkl')  # сохраняем лейберы для предсказаний

# 3. Формируем признаки и целевые переменные
X = train_df.drop(columns=['id', 'invoiceId', 'QtyShipped', 'UnitPrice'], errors='ignore')
y_qty = train_df['QtyShipped']
y_price = train_df['UnitPrice']

# 4. Сохраняем список признаков
features_used_in_training = X.columns
joblib.dump(features_used_in_training, 'features_used_in_training.pkl')

# 5. Разделение данных
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_qty, test_size=0.2, random_state=42
)

# Проверка размеров
print(f"Размер X_train: {X_train.shape}")
print(f"Размер y_train (QtyShipped): {y_train.shape}")
print(f"Размер y_price: {y_price.shape}")

# 6. Обучение моделей
model_qty = lgb.LGBMRegressor()
model_qty.fit(X_train, y_train)

model_price = lgb.LGBMRegressor()
model_price.fit(X_train, y_price.loc[X_train.index])  # убедитесь, что размеры совпадают

# 7. Создаём папку 'models', если её нет
os.makedirs('models', exist_ok=True)

# 8. Сохраняем модели
joblib.dump(model_qty, 'models/model_qty.pkl')
joblib.dump(model_price, 'models/model_price.pkl')

print("Модели обучены и успешно сохранены.")
