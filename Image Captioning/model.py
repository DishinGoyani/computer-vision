import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        
        # check if GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set hidden dimensional and creating embeding for fix size input for RNN
        self.hidden_dim = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM cell for RNN
        self.lstm = nn.LSTM(embed_size, self.hidden_dim, batch_first=True)
        
        # Converting RNN output to vocab size features to get probability for each word in vocabulary
        self.fc = nn.Linear(self.hidden_dim, vocab_size)
    
    def init_hidden(self, n_layers, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(n_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(n_layers, batch_size, self.hidden_dim).to(self.device))
    
    def forward(self, features, captions):
        
        # Initialize the hidden state
        n_layers = 1                             # number of LSTM layer
        batch_size = features.shape[0]           # features is of shape (batch_size, embed_size)
        
        self.hidden = self.init_hidden(n_layers, batch_size) # initializing hidden states
        
        # creating embeding for diven captions
        captions = captions[:, :-1]              # discard <end> token
        embed_captions = self.embedding(captions)
        
        # Concatenates image features (CNNEncode output) and captions
        embed_input = torch.cat((features.unsqueeze(1), embed_captions), dim=1)
        
        # get the output and hidden state by passing the lstm over our captions embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embed_input, self.hidden)
        
        # converting embedding to vocab size output
        output = self.fc(lstm_out)
        
        return output
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        
        for i in range(max_len):
            # predict next word
            output, states = self.lstm(inputs, states)
            output = self.fc(output)
            
            # get word index with highest probability
            _, predicted = output.max(2)
            tokens.append(predicted.item())
            
            # embedding predicted word for next input
            inputs = self.embedding(predicted)
                
        return tokens