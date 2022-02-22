import time
import torch
import data
from torch import nn
import math


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_data, seq_len):
        self.data = tensor_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        return self.data[i:i+self.seq_len], self.data[i+self.seq_len]


class FNNModel(nn.Module):
    def __init__(self, n_token, n_emb, n_hidden, seq_len):
        super().__init__()
        self.n_token = n_token
        self.n_emb = n_emb 
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.encoder = nn.Embedding(n_token, n_emb)
        self.hidden = nn.Linear(n_emb*seq_len, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_token)
    
    def forward(self, input):
        emb = self.encoder(input)
        batch_size = emb.shape[0]
        emb = emb.view(batch_size, -1)
        out = self.hidden(emb)
        out = torch.tanh(out)
        decoded = self.decoder(out)
        decoded = nn.functional.log_softmax(decoded, dim=1)
        return decoded
        
def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0 and i > 0:
            cur_loss = total_loss/100
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches |  ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i, len(train_loader),
                      elapsed * 1000/100, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate():
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
        return total_loss/len(val_loader)


###################################################
TRAIN_BATCH = 1024
EVAL_BATCH = 10000
SAVE_DIR = "model.pt"
EPOCH = 20
SEQ_LEN = 10
####################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data_dir = './data/wikitext-2'
    corpus = data.Corpus(data_dir)
    N_TOKENS = len(corpus.dictionary)
    train_loader = torch.utils.data.DataLoader(SequenceDataset(
        corpus.train, seq_len=SEQ_LEN), batch_size=TRAIN_BATCH)
    val_loader = torch.utils.data.DataLoader(SequenceDataset(
        corpus.valid, seq_len=SEQ_LEN), batch_size=EVAL_BATCH)
    test_loader = torch.utils.data.DataLoader(SequenceDataset(
        corpus.test, seq_len=SEQ_LEN), batch_size=EVAL_BATCH)
    model = FNNModel(
        n_token=N_TOKENS,
        n_emb=200,
        n_hidden=200,
        seq_len=SEQ_LEN,
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Training Starts")
    best_val_loss = None
    count = 0
    for epoch in range(1, EPOCH+1):
        print("-"*89)
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print("-"*89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(SAVE_DIR, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            count = 0
        else:
            count += 1
        if count > 5:
            print(f'Early Stop at epoch {epoch}')
            break

    with open(SAVE_DIR, 'rb') as f:
        model = torch.load(f).to(device)
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i, (X ,y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                total_loss += loss.item()
            test_loss = total_loss/len(test_loader)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
