import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import configparser
import ast
from tqdm import tqdm
import numpy as np
import torch

from models.SASRec import *
from datasets.movielens_sasrec import *
from datasets.kmrd_sasrec import *
from utils.evaluation import *
from utils.EarlyStopper import *


# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, 
        help="Random Seed.")
    parser.add_argument('--epochs', type=int, default=200,
        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048,
        help='Number of batch size for training.')
    parser.add_argument('--dataset', default='ml-1m',
        help='kmrd-2m, ml-1m or ml-20m.')
    parser.add_argument('--preprocessed_dataset', action='store_true', default=False,
        help=".")
    parser.add_argument('--model', default='sasrec',
        help="sasrec.")
    parser.add_argument('--pretrained_model', action='store_true', default=False,
        help=".")
    parser.add_argument("--top_k", type=int, default=10, 
        help="Compute metrics@top_k.")
    parser.add_argument("--num_workers", type=int, default=4,
        help="Number of cores for perform multi-process data loading")
    parser.add_argument('--no_cuda', action='store_true', default=False,
        help='Disables CUDA training.')
    parser.add_argument("--gpu", type=str, default="0",
        help="GPU Card ID.")
    
    return parser.parse_args()

def get_data(name):
    if name == 'ml-1m':
        return MovieLens1M_Data(path=config_dataset['path'],
                                min_rating=int(config_dataset['min_rating']),
                                max_len=int(config_dataset['max_len']),
                                num_neg_test=int(config_dataset['num_neg_test']))
    elif name == 'ml-20m':
        return MovieLens20M_Data(path=config_dataset['path'],
                                 min_rating=int(config_dataset['min_rating']),
                                 max_len=int(config_dataset['max_len']),
                                 num_neg_test=int(config_dataset['num_neg_test']))
    elif name == 'kmrd-2m':
        return KMRD2M_Data(path=config_dataset['path'],
                           min_rating=int(config_dataset['min_rating']),
                           max_len=int(config_dataset['max_len']),
                           num_neg_test=int(config_dataset['num_neg_test']))    
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, num_users, num_movies, device):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'sasrec':
        return SASRec(num_users,
                      num_movies,
                      hidden_units=int(config_model['hidden_units']),
                      num_heads=int(config_model['num_heads']),
                      num_blocks=int(config_model['num_blocks']),
                      max_len=int(config_dataset['max_len']),
                      dropout=float(config_model['dropout']),
                      device=device
                      )
    
def train(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = 0
    total_batch = len(data_loader)
    for u, seq, pos, neg in data_loader:
        u = u.to(device)
        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
        optimizer.zero_grad()
        indices = torch.where(pos != 0) # ignore padding
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        if float(config_model['l2_emb']) > 0:
            for param in model.item_emb.parameters(): loss += float(config_model['l2_emb']) * torch.norm(param)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / total_batch

def test(model, data_loader, total_users, device):
    model.eval()
    metrics_k = np.zeros(3, dtype=float) # map, ndcg, hit
    with torch.no_grad():
        for u, seq, item_idx in data_loader:
            u = u.to(device)
            seq = seq.to(device)
            item_idx = item_idx.to(device)
            predictions = -model.predict(u, seq, item_idx) # - for 1st argsort DESC
            for prediction in predictions:
                rank = prediction.argsort().argmin().item() # rank of index 0
                metrics_k += get_metrics_k_from_rank(rank, args.top_k)
                
    return metrics_k / total_users


if __name__ == '__main__':
    # set device and parameters
    args = parse_args()
    args.num_workers = args.num_workers * torch.cuda.device_count()
    config = configparser.ConfigParser()
    config.read('../config/dataset.ini')
    config_dataset = config[args.dataset]
    config.read('../config/model.ini')
    config_model = config[args.model]
    
    device = torch.device("cuda:"+args.gpu if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"device: {device}, model: {args.model}, dataset: {args.dataset}, negative_sampling: {args.negative_sampling}")
        
    data = get_data(args.dataset)
    train_dataset = data.get_train_dataset()
    valid_dataset = data.get_valid_dataset()
    test_dataset = data.get_test_dataset()
    print(f"train length: {len(train_dataset)}, valid length: {len(valid_dataset)}, test length: {len(test_dataset)}")
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers
                                                    )
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.num_workers
                                                   )
    
    model = get_model(args.model, train_dataset.num_users, train_dataset.num_movies, device).to(device)
    # print(f"model: \n {model}")
    # print("="*50)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    criterion = torch.nn.BCEWithLogitsLoss().to(device) # torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=float(config_model['learning_rate']),
                                betas=ast.literal_eval(config_model['betas'])
                                )
    
    early_stopper = EarlyStopper(num_trials=2, direction='maximize')
    tk0 = tqdm(range(1, args.epochs+1), smoothing=0, mininterval=1.0)
    for epoch in tk0:
        loss = train(model, optimizer, train_data_loader, criterion, device)
        tk0.set_postfix(loss=loss)
        if epoch % 10 == 0:
            metrics = test(model, valid_data_loader, data.num_users, device)
            print(f"[Valid] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, HR@K:{metrics[2]:.3f}")
            if not early_stopper.is_continuable(model, metrics[1]):
                print(f'[Valid] Best nDCG@K: {early_stopper.best_metric:.3f}')
                break
    else: 
        print(f'[Valid] Best nDCG@K: {early_stopper.best_metric:.3f}')
            
    metrics = test(model, test_data_loader, data.num_users, device)
    print(f"[Test] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, HR@K:{metrics[2]:.3f}")