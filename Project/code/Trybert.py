import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np

# Updated Dummy dataset
class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tweet_id, tweet_text = line.strip().split('\t')
                self.data.append((tweet_id, tweet_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Transformer-based encoder
class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval() 
       
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get a fixed-size representation
        return embeddings

# Example usage
def LoadTweet(obj):
    # Path to your text file
    file_path = "C:\\Users\\priya\\Documents\\ProjectData\\" + obj + "\\source_tweets.txt"

    dataset = MyDataset(file_path)

    embeddings_list = []
    result_list = []
    # Create transformer encoder
    transformer = TransformerEncoder()
    linear_layer = nn.Linear(768, 100)
    transformer.eval()

    # Encode sequences
    for tweet_ids, tweet_texts in dataset:
        embeddings = transformer(tweet_texts)
        embeddings = embeddings.view(1, -1)
        
        embeddings = linear_layer(embeddings)
        embeddings_list.append(embeddings)
    embeddings_array = np.vstack([emb.detach().numpy() for emb in embeddings_list])

    for i in range(len(dataset)):
        result_list.append({'tweet_id': dataset[i][0], 'embedding': embeddings_array[i]})
   
    print(len(result_list))
    save_path = "C:\\Users\\priya\\Documents\\ProjectData\\" + obj + "\\result.pt"
    torch.save(result_list, save_path)

data = LoadTweet("Twitter16")
