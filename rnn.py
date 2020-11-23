import torch
import torch.nn as nn

def rnn_preprocessing(data, test=False):
  """rnn_preprocessing

  :param data: The dataset to be converted into a vectorized format
  :type data: List[Tuple[List[str], int]] or List[List[str]]
  
  :param test: A list of vector representations of the input or pairs of vector
		representations with expected output
  :type test: List[Tuple[torch.Tensor, int]] or List[torch.Tensor]
  """
  # Do some preprocessing similar to convert_to_vector_representation
  # For the RNN, remember that instead of a single vector per training
  # example, you will have a sequence of vectors where each vector
  # represents some information about a specific token.
  vectorized_data = []
  max_length = 40
  if test:
    for doc in data:
      sentence = []
      for word in doc:
        if len(sentence) < max_length:
          index = word2index.get(word, word2index[UNK])
          sentence.append(index)
      sent_len = len(sentence)
      pad = []
      for i in range(max_length - sent_len):
        pad.append(word2index[UNK])
      vectorized_data.append(torch.tensor(pad+sentence))

  else:
    for doc, y in data:
      sentence = []
      for word in doc:
        if len(sentence) < max_length:
          index = word2index.get(word, word2index[UNK])
          sentence.append(index)
      sent_len = len(sentence)
      pad = []
      for i in range(max_length - sent_len):
        pad.append(word2index[UNK])
      vectorized_data.append((torch.tensor(pad+sentence), y))

  return vectorized_data

class RNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, drop_rate=0.3): # Add relevant parameters
    super(RNN, self).__init__()
    # Fill in relevant parameters
    self.hidden_dim = hidden_dim

    # define the layers
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim, 2, batch_first=True)
    self.W = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(drop_rate)
    self.output_dim = output_dim

    # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
    self.softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

def compute_Loss(self, predicted_vector, gold_label):
  return self.loss(predicted_vector, gold_label)	

def hidden0(self, inputs):
  batch_dim = inputs.size(0)
  h0 = torch.zeros(1, batch_dim, self.hidden_dim) 
  return h0

def forward(self, inputs):
  # begin code
  inputs = inputs.long()	
  embed = self.embedding(inputs) # embed has dimensions (batch_size, seq_length, embedding_dim)
  out, hn = self.rnn(embed) # out has dimensions (batch_size, seq_length, hidden_dim)
  output = out[:, -1, :]  # output has dimensions (batch_size, hidden_dim)
  output = self.dropout(output)
  z = self.W(output) # z has dimensions (batch_size, output_dim)
  # connect the layers
  predicted_vector = self.softmax(z) # remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
  return predicted_vector

def load_model(self, save_path):
  self.load_state_dict(torch.load(save_path))

def save_model(self, save_path):
  torch.save(self.state_dict(), save_path)
