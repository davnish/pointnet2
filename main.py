import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import Pointnet2Seg
from dataset import Dales
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse
import os
torch.manual_seed(42)

#Training the model
def train_loop(loader, see_batch_loss = False):
    model.train()
    total_loss = 0
    y_true = []
    y_preds = []
    for batch, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device).squeeze()

        logits = model(data)

        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        preds = logits.max(dim = -1)[1].view(-1)

        y_true.extend(label.view(-1).cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())
        
        if see_batch_loss:
            if batch%batch_eval_inter == 0:
                print(f'Batch_Loss_{batch} : {loss.item()}')

    return total_loss/len(loader), accuracy_score(y_true, y_preds), balanced_accuracy_score(y_true, y_preds)
        
@torch.no_grad()
def test_loop(loader, loss_fn, model, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_preds = []
    for data, label in loader:
        data, label = data.to(device), label.to(device).squeeze()
        # data = data.transpose(1,2)

        logits = model(data)

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))
        
        total_loss+=loss.item()
        preds = logits.max(dim = -1)[1].view(-1)
        
        y_true.extend(label.view(-1).cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())
        # np.savez('raw.npz', data1 = data.cpu(), y_true1 = y_true, y_preds1 = y_preds)
        # break
    # print(f'val_loss: {total_loss/len(test_loader)}, val_acc: {accuracy_score(y_true, y_preds)}')  
    return total_loss/len(loader), accuracy_score(y_true, y_preds), balanced_accuracy_score(y_true, y_preds), y_preds

if __name__ == '__main__':

    # Hyperparameter----

    grid_size = 25 # The size of the grid from 500mx500m 
    points_taken = 4096 # Points taken per each grid 
    batch_size = 8
    lr = 1e-4
    epoch = 100
    eval_train_test = 10
    n_embd = 64
    step_size = 50 # Reduction of Learning at how many epochs
    batch_eval_inter = 100
    dropout = 0.3
    # eval_test = 10

    # ------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type = float, default= lr)
    parser.add_argument('--epoch', type = int, default = epoch)
    parser.add_argument('--step_size', type = int, default = step_size)
    parser.add_argument('--model_name', default = '42')
    parser.add_argument('--batch_size', type = int,default = batch_size)
    parser.add_argument('--points_taken', type = int, default = points_taken)
    parser.add_argument('--grid_size', type = int, default = grid_size)


    args = parser.parse_args()


    # Setting Device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Splitting the data
    _dales = Dales(device, args.grid_size, args.points_taken, partition='train')
    print("File Read Complete")
    train_dataset, test_dataset = random_split(_dales, [0.7, 0.3])

    # Loading the data
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last=True)

    # Initialize the model
    model = Pointnet2Seg()

    # loss, Optimizer, Scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = 0.9)
    model = model.to(device)



    print("Running Epochs")
    print(f'{device = }, {args.grid_size = }, {args.points_taken = }, {args.epoch = }, {n_embd = }, {args.batch_size = }, {args.lr = }')
    start = time.time()
    for _epoch in range(1, args.epoch+1): 
        train_loss, train_acc, bal_avg_acc = train_loop(train_loader)
        scheduler.step()
        if _epoch%eval_train_test==0:
            val_loss, val_acc, bal_val_acc, _ = test_loop(test_loader, loss_fn, model, device)
            print(f'Epoch {_epoch} | lr: {scheduler.get_last_lr()}:\n train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | bal_train_acc: {bal_avg_acc:.4f}\n val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | bal_val_acc: {bal_val_acc:.4f}')
        


        # break
        
    end = time.time()

    print(f'Total_time: {end-start}')

    if not os.path.exists(os.path.join("model", "best")):
        os.makedirs(os.path.join("model", "best"))
    torch.save(model.state_dict(), os.path.join("model", "best", f"model_{args.model_name}.pt"))
    print(f"Model Saved at {args.epoch} epochs, named: model_{args.model_name}")