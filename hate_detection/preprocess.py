import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TweetsDataset(Dataset):
    def __init__(self, tokenizer, tweets, labels, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, item):
        tweet = self.tweets[item]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoding['input_ids'] = encoding['input_ids'].flatten()
        encoding['attention_mask'] = encoding['attention_mask'].flatten()
        if self.labels:
            encoding["labels"] = torch.tensor(self.labels[item], dtype=torch.long)
        return encoding
    
    def __len__(self):
        return len(self.tweets)


def create_data_loader(tokenizer, tweets, labels=None, max_len=512, batch_size=1):
    dataset = TweetsDataset(
        tweets = tweets,
        labels = labels, 
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )