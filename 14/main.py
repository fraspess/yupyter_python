import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine = load_wine()

# Перетворення у DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Ознаки та цільова змінна
X = df
y = wine.target

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pca_2 = PCA(n_components=2)

X_train_pca_2 = pca_2.fit_transform(X_train)
X_test_pca_2 = pca_2.transform(X_test)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_train_pca_2[:, 0],
    X_train_pca_2[:, 1],
    c=y_train,
    cmap='viridis'
)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components Visualization')

plt.colorbar(scatter, label='Target Class')

plt.show()

components_list = [2, 5, 10]
accuracy_results = []

for n in components_list:
    pca = PCA(n_components=n)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_results.append(accuracy)

    print(f'Кількість компонент: {n}')
    print(f'Accuracy: {accuracy:.4f}')
    print('-' * 40)


plt.figure(figsize=(8, 5))

plt.plot(
    components_list,
    accuracy_results,
    marker='o'
)

plt.xlabel('Number of PCA Components')
plt.ylabel('Classification Accuracy')
plt.title('PCA Accuracy vs Components')

plt.grid(True)

plt.savefig('pca_accuracy_results.png')

plt.show()