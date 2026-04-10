import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=10000, solver='lbfgs')

selector = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=1
)

print("Запуск процесу...")
selector.fit(X_scaled, y)

print(f"Початкова кількість ознак: {X.shape[1]}")
print(f"Оптимальна кількість ознак: {selector.n_features_}")

feature_info = pd.DataFrame({
    'Назва ознаки': X.columns,
    'Обрано': selector.support_,
    'Ранг': selector.ranking_
})

feature_info['Важливість'] = np.abs(selector.estimator_.coef_[0]) if selector.n_features_ == X.shape[1] else 0

print("\nТОП-10 найбільш значущих ознак за ваговими коефіцієнтами:")
print(feature_info.sort_values(by='Важливість', ascending=False).head(10)[['Назва ознаки', 'Важливість']])

plt.figure(figsize=(12, 6))
plt.title("Результати роботи RFECV", fontsize=14)
plt.xlabel("Кількість задіяних ознак", fontsize=12)
plt.ylabel("Точність класифікації", fontsize=12)

n_features_range = range(1, len(selector.cv_results_['mean_test_score']) + 1)
plt.plot(n_features_range, selector.cv_results_['mean_test_score'], marker='o', linestyle='-', color='#2ca02c', label='Середня точність на CV')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()