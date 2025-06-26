import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and clean data
df = pd.read_csv("../poetry_data.csv")
df.columns = df.columns.str.strip().str.lower()
df.dropna(inplace=True)

# Basic cleaning
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

df['hindi poetry'] = df['hindi poetry'].apply(preprocess)
df['english poetry'] = df['english poetry'].apply(preprocess)

# Vocabulary and tokenization
class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_count = {}

    def build_vocab(self, sentences):
        idx = 4
        for sent in sentences:
            for word in sent.split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def encode(self, sentence, max_len=20):
        tokens = [self.word2idx.get(word, 3) for word in sentence.split()]
        tokens = [1] + tokens[:max_len - 2] + [2]
        tokens += [0] * (max_len - len(tokens))
        return tokens

    def decode(self, tokens):
        return ' '.join([self.idx2word.get(token, "<unk>") for token in tokens if token not in [0, 1, 2]])

# Dataset
class PoetryDataset(Dataset):
    def __init__(self, hindi, english, vocab, max_len=20):
        self.hindi = hindi
        self.english = english
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.hindi)

    def __getitem__(self, idx):
        src = self.vocab.encode(self.hindi[idx], self.max_len)
        tgt = self.vocab.encode(self.english[idx], self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

# Build vocab
vocab = Vocab()
vocab.build_vocab(df['hindi poetry'])
vocab.build_vocab(df['english poetry'])

# Split data
train_hindi, val_hindi, train_eng, val_eng = train_test_split(df['hindi poetry'], df['english poetry'], test_size=0.1)
train_data = PoetryDataset(train_hindi.tolist(), train_eng.tolist(), vocab)
val_data = PoetryDataset(val_hindi.tolist(), val_eng.tolist(), vocab)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Model
class GPTLSTMHybrid(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.linear(output)
        return output

# Initialize model
vocab_size = len(vocab.word2idx)
model = GPTLSTMHybrid(vocab_size, embed_size=128, hidden_size=256).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "gpt_lstm_model.pth")

def translate_sentence(model, sentence, vocab, max_len=20):
    model.eval()
    with torch.no_grad():
        encoded = vocab.encode(sentence, max_len)
        input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)  # shape: [1, seq_len]

        outputs = model(input_tensor)  # shape: [1, seq_len, vocab_size]
        predicted_ids = torch.argmax(outputs, dim=-1).squeeze(0).tolist()

        translated_sentence = vocab.decode(predicted_ids)
        return translated_sentence

# Load model
model = GPTLSTMHybrid(vocab_size, embed_size=128, hidden_size=256).to(device)
model.load_state_dict(torch.load("gpt_lstm_model.pth"))

# Translate
input_poetry = "चाँदनी रात में सपने बुनता हूँ।"
translated = translate_sentence(model, input_poetry, vocab)
print("Hindi:", input_poetry)
print("Translated English:", translated)
