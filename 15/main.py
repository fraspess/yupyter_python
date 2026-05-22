import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN

X, _ = make_blobs(
    n_samples=600,
    centers=5,
    cluster_std=1.2,
    random_state=0
)

df = pd.DataFrame(X, columns=["income", "spending_score"])

plt.figure(figsize=(6, 6))
plt.scatter(df["income"], df["spending_score"])
plt.xlabel("income")
plt.ylabel("spending_score")
plt.title("Client Data")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
df["segment"] = kmeans.fit_predict(df[["income", "spending_score"]])

dbscan = DBSCAN(eps=1.5, min_samples=10)
df["cluster_dbscan"] = dbscan.fit_predict(df[["income", "spending_score"]])

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["segment"]
)
plt.xlabel("income")
plt.ylabel("spending_score")
plt.title("K-Means Segmentation")

plt.subplot(1, 2, 2)
plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["cluster_dbscan"]
)
plt.xlabel("income")
plt.ylabel("spending_score")
plt.title("DBSCAN Segmentation")

plt.tight_layout()
plt.savefig("client_segmentation_comparison.png")
plt.show()
