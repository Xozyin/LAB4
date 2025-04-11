import torch
import torch.nn as nn
import pandas as pd

# Класс нейронной сети
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()  # Для бинарной классификации — выдаёт от 0 до 1
        )

    def forward(self, X):
        return self.layers(X)

# Загружаем данные
df = pd.read_csv("dataset_simple.csv")

# Признаки — возраст и доход
X = torch.Tensor(df.iloc[:, 0:2].values)

# Целевая переменная — купит (1) или не купит (0)
y = torch.Tensor(df.iloc[:, 2].values.reshape(-1, 1))

# Размеры слоёв
inputSize = X.shape[1]  # 2 признака
hiddenSizes = 3
outputSize = 1

# Создаём сеть
net = NNet(inputSize, hiddenSizes, outputSize)

# Функция потерь и оптимизатор
lossFn = nn.BCELoss()  # Binary Cross Entropy — для задач бинарной классификации
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

# Обучение
epohs = 100
for i in range(epohs):
    pred = net(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Эпоха {i}: Ошибка = {loss.item():.4f}")

# Проверка точности после обучения
with torch.no_grad():
    predictions = net(X)
    predicted_labels = (predictions >= 0.5).float()
    accuracy = (predicted_labels == y).float().mean()
    errors = (predicted_labels != y).sum()

print(f"\nКоличество ошибок: {int(errors.item())}")
print(f"Точность: {accuracy.item() * 100:.2f}%")
