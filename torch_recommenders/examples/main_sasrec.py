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
        help='kmrd, ml-1m or ml-2m.')
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

def get_data():
    if args.dataset == 'ml-1m':
        return MovieLens1M_Data(path=config_dataset['path'],
                                min_rating=int(config_dataset['min_rating']),
                                max_len=int(config_dataset['max_len']),
                                num_neg_test=int(config_dataset['num_neg_test'])
                                )
    elif args.dataset == 'ml-20m':
        return MovieLens20M_Data(path=config_dataset['path'],
                                 min_rating=int(config_dataset['min_rating']),
                                 max_len=int(config_dataset['max_len']),
                                 num_neg_test=int(config_dataset['num_neg_test'])
                                 )
    else:
        raise ValueError('unknown dataset name: ' + args.dataset)

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
                      dropout_rate=float(config_model['dropout_rate']),
                      device=device
                      )
    
def train(model, optimizer, data_loader, criterion, device, log_interval=100):
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

def test(model, data_loader, num_users, device):
    model.eval()
    metrics_k = np.zeros(4, dtype=float) # map, ndcg, precision, recall
    with torch.no_grad():
        for u, seq, item_idx in data_loader:
            u = u.to(device)
            seq = seq.to(device)
            item_idx = item_idx.to(device)
            predictions = -model.predict(u, seq, item_idx) # - for 1st argsort DESC
            for prediction in predictions:
                rank = prediction.argsort().argmin().item() # rank of index 0
                metrics_k += get_metrics_k_from_rank(rank, args.top_k, len(data_loader))
                
    return metrics_k / num_users    


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
    print(f"device: {device}, model: {args.model}, dataset: {args.dataset}, preprocessed_dataset: {args.preprocessed_dataset}")
        
    if not args.preprocessed_dataset:
        data = get_data()
        train_dataset = data.get_train_dataset()
        valid_dataset = data.get_valid_dataset()
        test_dataset = data.get_test_dataset()
        torch.save(train_dataset, f"{config_dataset['preprocessed_path']}/train_dataset.pt")
        torch.save(valid_dataset, f"{config_dataset['preprocessed_path']}/valid_dataset.pt")
        torch.save(test_dataset, f"{config_dataset['preprocessed_path']}/test_dataset.pt")
    else:
        train_dataset = torch.load(f"{config_dataset['preprocessed_path']}/train_dataset.pt")
        valid_dataset = torch.load(f"{config_dataset['preprocessed_path']}/valid_dataset.pt")
        test_dataset = torch.load(f"{config_dataset['preprocessed_path']}/test_dataset.pt")
        
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
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.BCELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=float(config_model['learning_rate']),
                                betas=ast.literal_eval(config_model['betas'])
                                )
    
    tk0 = tqdm(range(1, args.epochs+1), smoothing=0, mininterval=1.0)
    for epoch in tk0:
        loss = train(model, optimizer, train_data_loader, criterion, device)
        tk0.set_postfix(loss=loss)
        if epoch % 20 == 0:
            val_metrics = test(model, valid_data_loader, train_dataset.num_users, device)
            tst_metrics = test(model, test_data_loader, train_dataset.num_users, device)
            print(f"[Valid] mAP@K: {val_metrics[0]:.3f}, nDCG@K: {val_metrics[1]:.3f}, Precision@K:{val_metrics[2]:.3f}, Recall@K: {val_metrics[3]:.3f}")
            print(f"[Test] mAP@K: {tst_metrics[0]:.3f}, nDCG@K: {tst_metrics[1]:.3f}, Precision@K:{tst_metrics[2]:.3f}, Recall@K: {tst_metrics[3]:.3f}")
            