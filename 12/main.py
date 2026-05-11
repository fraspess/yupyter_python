# ==========================================================
# Feedback Insight — Аналіз текстових відгуків клієнтів
# NLP: очищення тексту, токенізація, частотний аналіз, біграми
# ==========================================================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams

# ==========================================================
# 1. Завантаження необхідних ресурсів NLTK
# ==========================================================

nltk.download('punkt')
nltk.download('stopwords')

# ==========================================================
# 2. Створення прикладу CSV-файлу з відгуками
# ==========================================================

sample_reviews = [
    "The product quality is excellent and the customer service was amazing.",
    "I love this product, it works very well and exceeds expectations.",
    "The delivery was fast and the packaging was very secure.",
    "Customer support solved my issue quickly and professionally.",
    "The product is durable, reliable, and easy to use.",
    "Very good quality and outstanding performance.",
    "I am extremely satisfied with this purchase.",
    "The design is modern and the functionality is impressive.",
    "Great value for money and excellent build quality.",
    "The battery life is long and charging is very fast.",
    "Setup was simple and instructions were clear.",
    "The software interface is intuitive and user-friendly.",
    "Highly recommended for anyone looking for a dependable product.",
    "Performance is consistent and the product feels premium.",
    "This is one of the best products I have ever used."
]

# Зберігаємо у CSV
df_reviews = pd.DataFrame({"review": sample_reviews})
df_reviews.to_csv("customer_feedback.csv", index=False)

print("Файл customer_feedback.csv створено.")

# ==========================================================
# 3. Завантаження даних за допомогою pandas
# ==========================================================

df = pd.read_csv("customer_feedback.csv")

# ==========================================================
# 4. Очищення тексту
#    - нижній регістр
#    - видалення спецсимволів і цифр
# ==========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # залишаємо лише англійські літери
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_review"] = df["review"].apply(clean_text)

# ==========================================================
# 5. Базова статистика
# ==========================================================

df["word_count"] = df["clean_review"].apply(lambda x: len(x.split()))

num_reviews = len(df)
avg_length = df["word_count"].mean()

print(f"Кількість відгуків: {num_reviews}")
print(f"Середня довжина відгуку: {avg_length:.2f} слів")

# ==========================================================
# 6. Токенізація і видалення стоп-слів
# ==========================================================

stop_words = set(stopwords.words("english"))

# Токени до очищення
all_tokens_before = []

# Токени після очищення
all_tokens_after = []

for text in df["clean_review"]:
    tokens = word_tokenize(text)
    all_tokens_before.extend(tokens)

    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and len(token) >= 3
    ]

    all_tokens_after.extend(filtered_tokens)

print(f"Кількість токенів до очищення: {len(all_tokens_before)}")
print(f"Кількість токенів після очищення: {len(all_tokens_after)}")

# ==========================================================
# 7. Частотний словник
# ==========================================================

word_freq = Counter(all_tokens_after)

print("\nTop 15 Frequent Words:")
for word, count in word_freq.most_common(15):
    print(f"{word:<15} {count}")

# ==========================================================
# 8. Візуалізація топ-15 слів
# ==========================================================

top_words = word_freq.most_common(15)

words = [word for word, count in top_words]
counts = [count for word, count in top_words]

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], counts[::-1])  # reverse для правильного порядку
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title("Top 15 Frequent Words")
plt.tight_layout()
plt.savefig("feedback_word_freq.png", dpi=300)
plt.show()

print("Графік збережено як feedback_word_freq.png")

# ==========================================================
# 9. Аналіз біграм
# ==========================================================

# Генерація біграм
all_bigrams = list(bigrams(all_tokens_after))

# Підрахунок частоти
bigram_freq = Counter(all_bigrams)

# Топ-10 біграм
top_bigrams = bigram_freq.most_common(10)

# Створення DataFrame
bigram_df = pd.DataFrame({
    "Bigram": [' '.join(bg) for bg, count in top_bigrams],
    "Frequency": [count for bg, count in top_bigrams]
})

print("\nTop 10 Bigrams:")
print(bigram_df)

# Збереження у CSV
bigram_df.to_csv("feedback_bigrams.csv", index=False)

print("Файл feedback_bigrams.csv збережено.")

# ==========================================================
# 10. Додатково: збереження очищених даних
# ==========================================================

df.to_csv("customer_feedback_cleaned.csv", index=False)

print("Очищені дані збережено у customer_feedback_cleaned.csv")
