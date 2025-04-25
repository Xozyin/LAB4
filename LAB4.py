#Задание на лабораторную работу 4. Каневский Г.М. 23ВП2, Вариант 11

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv('dataset_simple.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.iloc[:, :2].values  # Первые два столбца: возраст и доход
y = df.iloc[:, 2].values   # Третий столбец: метка класса (0 или 1)

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)  # Признаки
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Целевые значения

# Нормализация данных вручную
mean = X_tensor.mean(dim=0)  # Среднее значение по каждому признаку
std = X_tensor.std(dim=0)    # Стандартное отклонение по каждому признаку

# Избегаем деления на ноль
std[std == 0] = 1e-8  # Заменяем нулевые значения на маленькое число

X_tensor_normalized = (X_tensor - mean) / std

# 2. Создание модели нейронной сети
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # Полносвязная сеть с одним скрытым слоем
        self.layers = nn.Sequential(
            nn.Linear(2, 8),  # Увеличили количество нейронов до 8
            nn.ReLU(),        # Функция активации ReLU
            nn.Linear(8, 1),  # Выходной слой (8 нейронов -> 1 выход)
            nn.Sigmoid()      # Сигмоида для бинарной классификации
        )
    
    def forward(self, x):
        return self.layers(x)

# Создаем экземпляр модели
model = SimpleClassifier()

# 3. Определяем функцию потерь и оптимизатор
criterion = nn.BCELoss()  # Бинарная кросс-энтропия для задачи классификации
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Используем Adam вместо SGD

# 4. Обучение модели
num_epochs = 500  # Количество эпох обучения
loss_history = []  # Для отслеживания ошибки

for epoch in range(num_epochs):
    # Прямой проход
    outputs = model(X_tensor_normalized)
    loss = criterion(outputs, y_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Сохраняем историю ошибок
    loss_history.append(loss.item())
    
    # Выводим ошибку каждые 50 эпох
    if (epoch + 1) % 50 == 0:
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Ошибка: {loss.item():.4f}')

# 5. Визуализация результатов
with torch.no_grad():
    predicted = model(X_tensor_normalized).numpy()
    predicted_classes = np.where(predicted >= 0.5, 1, 0)  # Классификация по порогу 0.5

# Визуализация данных
plt.figure(figsize=(8, 6))
x_min, x_max = X_tensor_normalized[:, 0].min() - 0.5, X_tensor_normalized[:, 0].max() + 0.5
y_min, y_max = X_tensor_normalized[:, 1].min() - 0.5, X_tensor_normalized[:, 1].max() + 0.5

# Создание сетки с увеличенным шагом
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Преобразование сетки в тензор PyTorch
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Предсказания модели
with torch.no_grad():
    grid_predictions = model(grid_tensor).numpy().reshape(xx.shape)

# Добавляем контурную карту
plt.contourf(xx, yy, grid_predictions, alpha=0.4, cmap=plt.cm.RdBu, levels=np.linspace(0, 1, 21))  # Градиентная заливка
plt.contour(xx, yy, grid_predictions, levels=[0.5], colors='black', linewidths=1.5)  # Четкая граница

# Визуализация точек данных
plt.scatter(X_tensor_normalized[y == 0, 0], X_tensor_normalized[y == 0, 1], color='red', label='Не купит', edgecolor='k')
plt.scatter(X_tensor_normalized[y == 1, 0], X_tensor_normalized[y == 1, 1], color='blue', label='Купит', edgecolor='k')

plt.title('Граница решений')
plt.xlabel('Возраст (нормализованный)')
plt.ylabel('Доход (нормализованный)')
plt.legend()
plt.show()

# 6. Оценка точности модели
accuracy = (predicted_classes.flatten() == y).mean()
print(f'Точность модели: {accuracy:.2f}')

# График изменения ошибки
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label='Ошибка', color='orange')
plt.title('Изменение ошибки')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.grid(True)  # Добавляем сетку
plt.legend()
plt.tight_layout()  # Улучшаем отображение графика
plt.show()
