import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils
from model import GCN, FCN
from evaluate import accuracy

# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.01,
        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=200,
        help='Number of epochs to train.')
    parser.add_argument('--num_hidden', type=int, default=16,
        help='Number of hidden units.')
    parser.add_argument("--model", type=str, default="GCN",
        help="Choose a model: GCN, FCN.")                
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='Disables CUDA training.')
    parser.add_argument("--gpu", type=str, default="0",
        help="GPU Card ID.")
    
    return parser.parse_args() 


if __name__ == "__main__":
    # set device and parameters
    args = parse_args()
    device = torch.device("cuda:"+args.gpu if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"device: {device}, model: {args.model}")
    
    tensorboard_name = f'{args.model}_ep:{args.epochs}_lr:{args.lr}_do:{args.dropout}_nh:{args.num_hidden}'
    writer = SummaryWriter(logdir="runs/" + tensorboard_name)
	
    # seed for Reproducibility
    utils.seed_everything(args.seed)
    
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()
    
    adj = adj.to(device)
    
    # set model and loss, optimizer
    if args.model == "GCN":
        model = GCN(num_feature=features.size(1),
                    num_hidden=args.num_hidden,
                    num_class=labels.unique().size(0),
                    adj_matrix=adj, # adj.to(device)
                    dropout=args.dropout)
    else:
        model = FCN(num_feature=features.size(1),
                    num_hidden=args.num_hidden,
                    num_class=labels.unique().size(0),
                    dropout=args.dropout)            

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
                           
    model = model.to(device)
    criterion = criterion.to(device)
    features = features.to(device)
    # adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)                       
    
    # train
    print("training...")
    best_acc, best_epoch = 0, 0
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        output = model(features) 
        train_loss = criterion(output[idx_train], labels[idx_train])
        train_acc = accuracy(output[idx_train], labels[idx_train]) 
        valid_loss = criterion(output[idx_val], labels[idx_val])
        valid_acc = accuracy(output[idx_val], labels[idx_val])
        
        train_loss.backward()
        optimizer.step()
        
        writer.add_scalar('loss/Train_loss', train_loss.data, epoch)
        writer.add_scalar('loss/Valid_loss', valid_loss.data, epoch)
        writer.add_scalar('Perfomance/Train_acc', train_acc, epoch)
        writer.add_scalar('Perfomance/Valid_acc', valid_acc, epoch)
        
        if epoch % 10 == 0:        
            print(f"Epoch: [{epoch:03d}/{args.epochs}]", end=" ")
            print(f"Train Loss: {train_loss.data:.4f}", end=" ")
            print(f"Val Loss: {valid_loss.data:.4f}", end=" ")
            print(f"Train ACC: {train_acc:.4f}", end=" ")
            print(f"Val ACC: {valid_acc:.4f}")
        
        if valid_acc > best_acc:
            best_acc, best_epoch = valid_acc, epoch
    
    writer.close()            
    t_total = time.time() - start_time
    print("Optimization Finished!")
    print(f"Total time elapsed: {t_total:.4f}s")
    print(f"End. Best epoch {best_epoch:03d}: ACC = {best_acc:.4f}")
    
    # test
    model.eval()
    output = model(features)
    test_loss = criterion(output[idx_test], labels[idx_test])
    test_acc = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results: loss= {test_loss:.4f} accuracy= {test_acc:.4f}")