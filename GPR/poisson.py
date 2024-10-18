import numpy as np
import matplotlib.pyplot as plt

# Параметри Пуассонівського процесу
lambda_rate = 3  # Параметр інтенсивності (кількість подій за одиницю часу)
T = 10           # Загальний час спостереження
N = 100          # Кількість точок для симуляції

# Генерація міжприходних інтервалів (експоненційно розподілені)
inter_arrival_times = np.random.exponential(1/lambda_rate, N)

# Генерація Пуассонівського процесу (накопичувальна сума міжприходних інтервалів)
arrival_times = np.cumsum(inter_arrival_times)

# Залишити тільки ті часи, які входять у загальний час T
arrival_times = arrival_times[arrival_times <= T]

# Генерація відповідних значень процесу (накопичувальний підрахунок подій)
process_values = np.arange(1, len(arrival_times) + 1)

# Побудова графіка Пуассонівського процесу
plt.figure(figsize=(10, 6))
plt.step(arrival_times, process_values, where='post', label='Poisson Process')
plt.title('Poisson Process')
plt.xlabel('Time $t$')
plt.ylabel('Number of Events')
plt.grid(True)
plt.legend()
plt.show()
