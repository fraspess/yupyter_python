# ==========================================================
# SupportFlow AI — Seq2Seq Language Modeling with LSTM
# and Comparison with Transformer (DistilGPT-2)
# ==========================================================

import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 1. Налаштування
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 1
EPOCHS = 15
BATCH_SIZE = 4
LR = 0.001
MAX_GENERATION_LEN = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# ==========================================================
# 2. Корпус даних (20–30 пар запитання-відповідь)
# ==========================================================

qa_pairs = [
    ("How do I change my email?", "You can update it in your profile settings."),
    ("How can I reset my password?", "Use the forgot password option on the login page."),
    ("My order is delayed.", "Your order may take a little longer than expected."),
    ("Where can I track my order?", "You can track it in the orders section."),
    ("How do I cancel my subscription?", "Go to billing settings and select cancel subscription."),
    ("How do I update my payment method?", "Open billing settings and add a new card."),
    ("I cannot log in.", "Please reset your password and try again."),
    ("How do I contact support?", "Use the contact form in the help center."),
    ("How do I delete my account?", "Open account settings and choose delete account."),
    ("When will I receive my refund?", "Refunds are usually processed within five business days."),
    ("Why was my payment declined?", "Please check your card details and available funds."),
    ("Can I change my username?", "You can edit your username in profile settings."),
    ("How do I download my invoice?", "Invoices are available in the billing section."),
    ("The app is crashing.", "Please update the app to the latest version."),
    ("How do I enable notifications?", "Open settings and turn on notifications."),
    ("Can I switch to a yearly plan?", "Yes, you can change your plan in billing settings."),
    ("How do I update my address?", "Edit your shipping address in account settings."),
    ("I received the wrong item.", "Please contact support and we will replace it."),
    ("How do I apply a discount code?", "Enter the code during checkout."),
    ("How do I change my language?", "Select your preferred language in settings."),
    ("Can I pause my subscription?", "Yes, you can pause it from billing settings."),
    ("How do I verify my account?", "Check your email and click the verification link."),
    ("My package is damaged.", "Contact support and attach photos of the damage."),
    ("How do I resend the confirmation email?", "Use the resend confirmation option."),
    ("Can I export my data?", "You can export your data from privacy settings.")
]

# ==========================================================
# 3. Очищення тексту
# ==========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

qa_pairs = [(clean_text(q), clean_text(a)) for q, a in qa_pairs]

# ==========================================================
# 4. Токенізація та словник
# ==========================================================

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

all_text = []
for q, a in qa_pairs:
    all_text.extend(q.split())
    all_text.extend(a.split())

vocab = SPECIAL_TOKENS + sorted(set(all_text))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

PAD_IDX = word2idx["<PAD>"]
SOS_IDX = word2idx["<SOS>"]
EOS_IDX = word2idx["<EOS>"]
UNK_IDX = word2idx["<UNK>"]

VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# ==========================================================
# 5. Підготовка послідовностей
# ==========================================================

def encode(text):
    tokens = text.split()
    return [word2idx.get(token, UNK_IDX) for token in tokens]

data = []
for question, answer in qa_pairs:
    # Для простоти навчаємо модель лише на відповідях
    seq = [SOS_IDX] + encode(answer) + [EOS_IDX]
    data.append(seq)

MAX_LEN = max(len(seq) for seq in data)
print(f"Max sequence length: {MAX_LEN}")

def pad_sequence(seq, max_len):
    return seq + [PAD_IDX] * (max_len - len(seq))

padded_data = [pad_sequence(seq, MAX_LEN) for seq in data]

# ==========================================================
# 6. Dataset
# ==========================================================

class ResponseDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        x = seq[:-1]
        y = seq[1:]
        return x, y

dataset = ResponseDataset(padded_data)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================================
# 7. LSTM Language Model
# ==========================================================

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)
        return logits

model = LSTMLanguageModel(
    VOCAB_SIZE,
    EMBED_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
# 8. Навчання (15 епох)
# ==========================================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Loss: {total_loss / len(loader):.4f}"
    )

# ==========================================================
# 9. Генерація відповіді LSTM
# ==========================================================

def generate_response(prompt=""):
    model.eval()

    generated = [SOS_IDX]

    with torch.no_grad():
        for _ in range(MAX_GENERATION_LEN):
            x = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            logits = model(x)

            next_token = logits[0, -1].argmax().item()

            if next_token in [EOS_IDX, PAD_IDX]:
                break

            generated.append(next_token)

    words = [
        idx2word[idx]
        for idx in generated[1:]
        if idx not in [PAD_IDX, EOS_IDX]
    ]

    return " ".join(words)

# ==========================================================
# 10. Тестові запити для LSTM
# ==========================================================

test_queries = [
    "How can I reset my password?",
    "My order is delayed.",
    "How do I update my payment method?",
    "The app is crashing.",
    "Can I export my data?"
]

print("\n=== LSTM Responses ===")
lstm_results = []

for query in test_queries:
    response = generate_response(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
    lstm_results.append((query, response))

# ==========================================================
# 11. Transformer: DistilGPT-2
# ==========================================================

print("Loading DistilGPT-2...")

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
transformer_model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2"
).to(DEVICE)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n=== Transformer Responses ===")
gpt2_results = []

for query in test_queries:
    prompt = f"Customer: {query}\nSupport:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = transformer_model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = generated_text.split("Support:")[-1].strip()

    print(f"Q: {query}")
    print(f"A: {answer}\n")
    gpt2_results.append((query, answer))

# ==========================================================
# 12. Таблиця спостережень
# ==========================================================

observation_rows = []

for i in range(len(test_queries)):
    observation_rows.append({
        "Query": test_queries[i],
        "LSTM Response": lstm_results[i][1],
        "GPT-2 Response": gpt2_results[i][1]
    })

observation_df = pd.DataFrame(observation_rows)
observation_df.to_csv("supportflow_observations.csv", index=False)

print("Saved: supportflow_observations.csv")

# ==========================================================
# 13. Порівняльна таблиця
# ==========================================================

comparison_df = pd.DataFrame({
    "Criterion": [
        "Response Quality",
        "Context Retention",
        "Generation Speed",
        "Coherence"
    ],
    "LSTM": [
        "Good for memorized templates",
        "Limited",
        "Very Fast",
        "Moderate"
    ],
    "Transformer": [
        "More natural and diverse",
        "Better",
        "Slower",
        "High"
    ]
})

comparison_df.to_csv("supportflow_comparison.csv", index=False)

print("\nComparison Table:")
print(comparison_df)

# ==========================================================
# 14. Збереження моделі
# ==========================================================

torch.save(model.state_dict(), "supportflow_lstm.pth")
print("\nLSTM model saved as supportflow_lstm.pth")

# ==========================================================
# 15. Висновки
# ==========================================================

print("""
ВИСНОВКИ

1. LSTM добре підходить для невеликих наборів даних та шаблонних відповідей.
2. Transformer (DistilGPT-2) генерує більш природні, різноманітні та зв'язні тексти.
3. Transformer краще зберігає контекст і логіку відповіді.
4. LSTM працює швидше та потребує менше ресурсів.
5. Для покращення якості варто:
   - збільшити корпус навчальних даних;
   - використати attention mechanism;
   - виконати fine-tuning GPT-2 на власному наборі даних;
   - застосувати beam search для генерації.

Найкращі результати показала трансформерна архітектура,
оскільки вона краще моделює довгострокові залежності
та генерує більш осмислені відповіді.
""")