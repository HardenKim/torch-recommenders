import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import configparser
import ast
from tqdm import tqdm
import numpy as np
import torch

from datasets.movielens import *
from datasets.kmrd import *
from models.FM import *
from models.NMF import *
from models.Wide_Deep import *
from models.DeepFM import *
from models.FFM import *
from utils.evaluation import *
from utils.EarlyStopper import *


# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, 
        help="Random Seed.")
    parser.add_argument('--epochs', type=int, default=100,
        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2048,
        help='Number of batch size for training.')
    parser.add_argument('--dataset', default='ml-1m',
        help='kmrd-2m, ml-1m or ml-20m.')
    parser.add_argument('--negative_sampling', action='store_false', default=True,
        help=".")
    parser.add_argument('--model', default='ncf',
        help="fm, ffm, wd, dfm or nmf.")
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
    parser.add_argument("--save_dir", type=str, default="best_model",
        help=".")
    
    return parser.parse_args()

def get_data(name):
    if name == 'ml-1m':
        return MovieLens1M_Data(path=config_dataset['path'],
                                min_rating=int(config_dataset['min_rating']),
                                negative_sampling=args.negative_sampling,
                                num_neg=int(config_dataset['num_neg']),
                                num_neg_test=int(config_dataset['num_neg_test']))
    elif name == 'ml-20m':
        return MovieLens20M_Data(path=config_dataset['path'],
                                 min_rating=int(config_dataset['min_rating']),
                                 negative_sampling=args.negative_sampling,
                                 num_neg=int(config_dataset['num_neg']),
                                 num_neg_test=int(config_dataset['num_neg_test']))
    elif name == 'kmrd-2m':
        return KMRD2M_Data(path=config_dataset['path'],
                           min_rating=int(config_dataset['min_rating']),
                           negative_sampling=args.negative_sampling,
                           num_neg=int(config_dataset['num_neg']),
                           num_neg_test=int(config_dataset['num_neg_test']))
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, field_dims):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'fm':
        return FactorizationMachineModel(field_dims,
                                         embed_dim=int(config_model['embed_dim']))
    elif name == 'ffm':
            return FieldAwareFactorizationMachineModel(field_dims,
                                                       embed_dim=int(config_model['embed_dim']))
    elif name == 'wd':
        return WideAndDeepModel(field_dims,
                                embed_dim=int(config_model['embed_dim']), 
                                mlp_dims=ast.literal_eval(config_model['mlp_dims']), 
                                dropout=float(config_model['dropout']))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims,
                                             embed_dim=int(config_model['embed_dim']),
                                             mlp_dims=ast.literal_eval(config_model['mlp_dims']),
                                             dropout=float(config_model['dropout']))
    elif name == 'nmf':
        return NeuralMatrixFactorization(field_dims,
                                         embed_dim=int(config_model['embed_dim']),
                                         mlp_dims=ast.literal_eval(config_model['mlp_dims']),
                                         dropout=float(config_model['dropout']))
        
def train(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = 0
    total_batch = len(data_loader)
    for fields, target in data_loader:
        fields, target = fields.view(-1,2).to(device), target.view(-1).to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / total_batch

def test(model, data_loader, total_users, device):
    model.eval()
    metrics_k = np.zeros(3, dtype=float) # map, ndcg, hit
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.view(-1,2).to(device), target.view(-1).to(device)
            predictions = model(fields)
            _, indices = torch.topk(predictions, args.top_k)
            recommends = torch.take(fields[:, [1]], indices).cpu().tolist()
            gt_item = fields[0][1].item() # leave one-out evaluation has only one item per user
            metrics_k += get_metrics_k(gt_item, recommends)
            
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
    
    if args.negative_sampling:
        train_batch_size = args.batch_size // (int(config_dataset['num_neg']) + 1)
        test_batch_size = 1
    else:
        train_batch_size = args.batch_size
        test_batch_size = 1
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=train_batch_size,
                                                    shuffle=True, 
                                                    num_workers=args.num_workers)                                                    
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                    batch_size=test_batch_size, # (num_neg_test + 1)
                                                    shuffle=False, 
                                                    num_workers=args.num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,  
                                                    batch_size=test_batch_size, # (num_neg_test + 1)
                                                    shuffle=False, 
                                                    num_workers=args.num_workers)
    
    model = get_model(args.model, data.field_dims).to(device)
    # print(f"model: \n {model}")
    # print("="*50)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=float(config_model['learning_rate']),
                                 weight_decay=float(config_model['weight_decay']))
    
    early_stopper = EarlyStopper(num_trials=5, direction='maximize', save_path=f'{args.save_dir}/{args.model}.pt')
    # early_stopper = EarlyStopper(num_trials=2, direction='maximize')
    tk0 = tqdm(range(1, args.epochs+1), smoothing=0, mininterval=1.0)
    for epoch in tk0:
        loss = train(model, optimizer, train_data_loader, criterion, device)
        tk0.set_postfix(loss=loss)
        metrics = test(model, valid_data_loader, data.num_users, device)
        if not early_stopper.is_continuable(model, metrics[1]):
            break
        if epoch % 10 == 0:
            print(f"[Valid] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, HR@K:{metrics[2]:.3f}")
    print(f'[Valid] Best nDCG@K: {early_stopper.best_metric:.3f}')
    model = torch.load(f'{args.save_dir}/{args.model}.pt')
    metrics = test(model, test_data_loader, data.num_users, device)
    print(f"[Test] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, HR@K:{metrics[2]:.3f}")