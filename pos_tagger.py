import conllu
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import nltk

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

# taking input and doing cute shit out of that i think
model.load_state_dict(torch.load('./pos_tagger.pt', map_location = device))
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
  
  for element in ret:
    print(element[0], "\t", element[1])
