# -*- coding: utf-8 -*-
"""inlp ass 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uatJWV8XY17-jqBOX4JOp2Jj0YumzJa5
"""

# pip install conllu

import conllu
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

import regex as re

with open('./en_atis-ud-train.conllu', mode='r') as data:
  annotations = data.read()

sentences = conllu.parse(annotations)

training_data = []

for sentence in sentences:
  tokens = []
  tags = []

  for word in sentence:
    tokens.append(word['form'])
    tags.append(word['upos'])

  training_data.append((tokens, tags))

with open('./en_atis-ud-dev.conllu', mode='r') as data:
  annotations = data.read()

sentences = conllu.parse(annotations)

dev_data = []

for sentence in sentences:
  tokens = []
  tags = []

  for word in sentence:
    tokens.append(word['form'])
    tags.append(word['upos'])

  dev_data.append((tokens, tags))

with open('./en_atis-ud-test.conllu', mode='r') as data:
  annotations = data.read()

sentences = conllu.parse(annotations)

test_data = []

for sentence in sentences:
  tokens = []
  tags = []

  for word in sentence:
    tokens.append(word['form'])
    tags.append(word['upos'])

  test_data.append((tokens, tags))

def prepare_sequence(seq, to_index_map):
  indx = []
  
  for ele in seq:
    if ele not in to_index_map:
        ele = '<unk>'
    indx.append(to_index_map[ele])
  return torch.tensor(indx, dtype = torch.long )

total_data = training_data + dev_data + test_data

word_to_index_map = {}
tag_to_index_map = {}

for sent, tags in total_data:
  for word in sent:
    if word not in word_to_index_map:
      word_to_index_map[word] = len(word_to_index_map)

for sent, tags in total_data:
  for tag in tags:
    if tag not in tag_to_index_map:
      tag_to_index_map[tag] = len(tag_to_index_map)

word_to_index_map['<unk>'] = len(word_to_index_map)

print(word_to_index_map)
print(tag_to_index_map)

class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, target_size):
    super(LSTMTagger, self).__init__()

    self.hidden_dim = hidden_dim
    
    self.num_layers = num_layers

    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers)

    self.hidden2tag = nn.Linear(hidden_dim, target_size)
  
  def forward(self, sentence):
    embeds = self.word_embeddings(sentence)
    lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    tag_scores = F.log_softmax(tag_space, dim = 1)
    return tag_scores

#hyperparameters

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 10
LEARNING_RATE = 0.01
NUM_LAYERS = 2

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, len(word_to_index_map.keys()), len(tag_to_index_map.keys()))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

loss_function = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

print("Starting Training...")

best_valid_loss = float('inf')

for epoch in range(EPOCHS):

  # Training Loop
  train_loss = 0
  model.train()

  for sent, tags in training_data:
    optimizer.zero_grad()
    model.zero_grad()
    sentence_in = prepare_sequence(sent, word_to_index_map)
    targets = prepare_sequence(tags, tag_to_index_map)

    tag_scores = model(sentence_in)

    loss = loss_function(tag_scores, targets)

    loss.backward()
    optimizer.step()

    train_loss += loss.item()
  
  # Validation loop
  with torch.no_grad():
    val_loss = 0
    model.eval()

    for sent, tags in dev_data:
      sentence_in = prepare_sequence(sent, word_to_index_map)
      targets = prepare_sequence(tags, tag_to_index_map)

      tag_scores = model(sentence_in)

      loss = loss_function(tag_scores, targets)
      val_loss += loss.item()

  print(f'Epoch {epoch+1} \t Training Loss: {train_loss / len(training_data)} \t Validation Loss: {val_loss / len(dev_data)}')

  if (val_loss / len(dev_data)) < best_valid_loss:
    best_valid_loss = val_loss / len(dev_data)
    torch.save(model.state_dict(), 'pos_tagger.pt')

acc_scores = 0
precision_scores = 0
recall_scores = 0
f1_scores = 0

with torch.no_grad():
  for sent, tags in test_data:
    inputs = prepare_sequence(sent, word_to_index_map)
    targets = prepare_sequence(tags, tag_to_index_map)

    tag_scores = model(inputs)
    _ , indices = torch.max(tag_scores, 1)

    report_dict = classification_report(targets, indices, output_dict = True, zero_division = 0)

    acc_scores += report_dict['accuracy']
    precision_scores += report_dict['weighted avg']['precision']
    recall_scores += report_dict['weighted avg']['recall']
    f1_scores += report_dict['weighted avg']['f1-score']

    ret = []
    for i in range(len(indices)):
      for key, value in tag_to_index_map.items():
        if indices[i] == value:
          ret.append((sent[i], key))

print("Accuracy: ", acc_scores/len(test_data))
print("Precision: ", precision_scores/len(test_data))
print("Recall: ", recall_scores/len(test_data))
print("F1 Score: ", f1_scores/len(test_data))

# taking input and doing cute shit out of that i think
model.load_state_dict(torch.load('pos_tagger.pt', map_location = device))
taken_input = input("Enter a sentence: ")

with torch.no_grad():
  sentence = taken_input.lower()
  sentence = re.sub(r'\s*([,.?!;:"()—_\\])\s*', r' ', sentence)
  sent = word_tokenize(sentence)

  inputs = prepare_sequence(sent, word_to_index_map)

  tag_scores = model(inputs)
  _ , indices = torch.max(tag_scores, 1)

  ret = []
  for i in range(len(indices)):
    for key, value in tag_to_index_map.items():
      if indices[i] == value:
        ret.append((sent[i], key))
  
  print(ret)