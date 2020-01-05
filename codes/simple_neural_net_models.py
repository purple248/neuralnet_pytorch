import torch
import torch.nn.functional as F
import torch.nn as nn



##################################################################
#DENSE NET
##################################################################

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.n_feature = n_feature
        self.fc1 = nn.Linear(n_feature, n_hidden)  # hidden layer
        self.fc2 = nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = x.view(-1, self.n_feature)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



###################################################################
#RNN NET
###################################################################

class RNN_Net(nn.Module):
    def __init__(self, n_feature, n_output, hidden_dim, n_layers, sequence_len):
        super(RNN_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers  # Number of recurrent layers
        self.sequence_len = sequence_len
        self.n_feature = n_feature

        self.rnn = nn.RNN(n_feature, hidden_dim, n_layers, batch_first=True)
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.fc = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # if self.n_feature == 1:
        #     x = x.view(batch_size, self.sequence_len, self.n_feature)

        out, _ = self.rnn(x, hidden)
        out = out[:, -1, :]
        # rnn_out will of size [batch_size, seq_len, hidden_size]
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim) - for singal number y

        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden



###################################################################
#LSTM NET
###################################################################

class LSTM_Net(nn.Module):
    def __init__(self, n_feature, n_output, hidden_dim, n_layers, sequence_len):
        super(LSTM_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers  # Number of lstm layers
        self.sequence_len = sequence_len
        self.n_feature = n_feature

        self.lstm = nn.LSTM(n_feature, hidden_dim, n_layers, batch_first=True) #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        self.fc = nn.Linear(hidden_dim, n_output)

    def forward(self, x):

        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size) #if we do shuffle

        # if self.n_feature == 1:
        #     x = x.view(batch_size, self.sequence_len, self.n_feature)

        out, _ = self.lstm(input=x, hx=hidden)
        # rnn_out will of size [batch_size, seq_len, hidden_size]

        out = out[:, -1, :]
        out = self.fc(out)
        #out = self.fc(out.view(-1, self.hidden_dim))

        return out

    def init_hidden(self, batch_size):
        # this method generates the first hidden state of zeros to use in the forward pass
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return (hidden_state, cell_state)