# poems_generation.py

import os
import re
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
import pickle
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# Download the datasets
os.system("wget https://huggingface.co/datasets/Libosa2707/vietnamese-poem/resolve/main/poems_dataset.csv")
os.system("wget https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/resolve/main/poems_dataset.csv")

# Install necessary libraries
os.system("pip install underthesea")

# Define functions for saving and loading pickle files
def save_pkl(save_object, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(load_file):
    with open(load_file, 'rb') as f:
        output = pickle.load(f)
    return output

# Load the dataset
data = pd.read_csv("poems_dataset.csv")

# Display the first few rows of the dataset
print(data.head())

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(data, test_size=0.3, random_state=42)

# Prepare the documents
train_documents = [doc for doc in train_df['content'].tolist()]
val_documents = [doc for doc in val_df['content'].tolist()]

# Vocabulary class definition
class Vocabulary:
    def __init__(self):
        self.word2id = dict()
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

        self.word2id['<pad>'] = self.pad_id
        self.word2id['<unk>'] = self.unk_id
        self.word2id['<s>'] = self.sos_id
        self.word2id['</s>'] = self.eos_id

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def lookup_tokens(self, word_indexes: list):
        return [self.id2word[word_index] for word_index in word_indexes]

    def add(self, word):
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
        indicies_corpus = list()
        for document in tqdm(tokenized_corpus):
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)), dtype=torch.long)
            indicies_corpus.append(indicies_document)
        return indicies_corpus

    def tensor_to_corpus(self, tensor):
        corpus = list()
        for indicies in tqdm(tensor):
            document = list(map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)
        return corpus

    @staticmethod
    def tokenize_corpus(corpus):
        print("Tokenize the corpus...")
        tokenized_corpus = list()
        for document in tqdm(corpus):
            tokenized_document = ['<s>'] + re.findall(r'(\w+|[^\w\s]|\S+|\n)', document) + ['</s>']
            tokenized_corpus.append(tokenized_document)
        return tokenized_corpus

    @classmethod
    def from_documents(cls, documents):
        words = set(word for doc in documents for word in re.findall(r'\w+|\S|\n', doc))
        vocab = cls()
        for w in words:
            vocab.add(w)
        return vocab

    @classmethod
    def from_pretrained(cls, save_dir):
        with open(os.path.join(save_dir, "vocab.pkl"), 'rb') as file:
            pretrained_vocab = pickle.load(file)
        return cls.init_vocab_from_pretrained(pretrained_vocab)

    @staticmethod
    def init_vocab_from_pretrained(pretrained_vocab):
        vocab = Vocabulary()
        vocab.word2id.update(pretrained_vocab)
        vocab.id2word = {v: k for k, v in vocab.word2id.items()}
        return vocab

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vocab.pkl"), 'wb') as file:
            pickle.dump(self.word2id, file)

# Initialize Vocabulary
vocab = Vocabulary.from_documents(train_documents)

# PoemGenerationDataset class definition
class PoemGenerationDataset(Dataset):
    def __init__(self, documents, vocab, max_length=None):
        self.vocab = vocab
        self.sos_idx = vocab["<s>"]
        self.eos_idx = vocab["</s>"]
        self.pad_idx = vocab["<pad>"]
        self.documents = documents
        self.max_length = max_length
        self.tokenized_documents = self.vocab.tokenize_corpus(self.documents)
        self.tensor_data = self.vocab.corpus_to_tensor(self.tokenized_documents, is_tokenized=True)

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx]

    def shift_right(self, input_ids, pad_token=0):
        padding_column = torch.full_like(input_ids[:, :1], pad_token)
        shifted_ids = torch.cat([padding_column, input_ids[:, :-1]], dim=-1)
        return shifted_ids

    def collate_fn(self, examples):
        examples = sorted(examples, key=lambda e: len(e), reverse=True)
        if self.max_length is not None:
            examples = [torch.cat([e[:self.max_length-1], torch.tensor([self.eos_idx])]) for e in examples]
        docs = [e for e in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True, padding_value=self.pad_idx)
        labels = input_ids.clone()
        input_ids = self.shift_right(input_ids, pad_token=self.pad_idx)
        return {"inputs": input_ids, "labels": labels}

# Initialize datasets and dataloaders
train_dataset = PoemGenerationDataset(train_documents, vocab, max_length=512)
val_dataset = PoemGenerationDataset(val_documents, vocab, max_length=512)
train_dataloader = DataLoader(train_dataset, batch_size=5, collate_fn=train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=5, collate_fn=val_dataset.collate_fn)

# LSTMForPoemGeneration model definition
class LSTMForPoemGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_idx)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size,
                            config.hidden_size,
                            num_layers=config.num_layers,
                            dropout=config.dropout,
                            batch_first=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, labels=None):
        embeds = self.dropout(self.embedding(input_ids))
        bn_output = self.batch_norm(embeds.permute(0,2,1)).permute(0,2,1)
        output, (hidden, cell) = self.lstm(bn_output)
        logits = self.lm_head(self.dropout(output))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, logits.size(2)), labels.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0):
        self.to(input_ids.device)
        self.eval()
        current_length = input_ids.size(1)
        while current_length < max_length:
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            current_length += 1
            if next_token_id.item() == vocab.eos_id:
                break
        return input_ids

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.config, os.path.join(save_dir, "config.pt"))
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.pt"))

    @classmethod
    def from_pretrained(cls, saved_dir):
        config = torch.load(os.path.join(saved_dir, "config.pt"))
        model = cls(config)
        state_dict = torch.load(os.path.join(saved_dir, "pytorch_model.pt"), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

# Configuration for the LSTM model
@dataclass
class LSTMConfig:
    vocab_size: int
    padding_idx: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.3

# Initialize and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_config = LSTMConfig(vocab_size=len(vocab), padding_idx=vocab.pad_id, hidden_size=128, num_layers=2)
model = LSTMForPoemGeneration(model_config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader), 1):
        optimizer.zero_grad()
        inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
        _, loss = model(inputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Training Loss after epoch {epoch}: {running_loss/len(train_dataloader)}")
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader), 1):
            inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
            _, loss = model(inputs, labels)
            val_loss += loss.item()
    print(f"Validation Loss after epoch {epoch}: {val_loss/len(val_dataloader)}")

# Save the model and vocabulary
model.save_pretrained("poem_model")
vocab.save_pretrained("poem_model")

# Generate sample text
input_sentence = "Mặt trời mọc ở đằng đông"
input_ids = vocab.corpus_to_tensor([input_sentence])[0].unsqueeze(0).to(device)
output = model.generate(input_ids, max_length=100, temperature=0.8)
output_text = " ".join(vocab.lookup_tokens(output.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
print(output_text)
