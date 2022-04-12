import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import configparser
import ast
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset
from datasets.movielens import *
from models.FM import *
from models.NCF import *
from models.Wide_Deep import *
from models.NFM import *
from models.DeepFM import *
from models.xDeepFM import *
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
    parser.add_argument('--model', default='ncf',
        help="fm, ncf, wd, nfm, dfm or xdfm.")
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

def get_dataset():
    if args.dataset == 'ml-1m':
        return MovieLens1M_Dataset(path=config_dataset['path'],
                                   min_rating=int(config_dataset['min_rating']),
                                   num_neg=int(config_dataset['num_neg']),
                                   num_neg_test=int(config_dataset['num_neg_test'])
                                   )
    elif args.dataset == 'ml-20m':
        return MovieLens20M_Dataset(path=config_dataset['path'],
                                    min_rating=int(config_dataset['min_rating']),
                                    num_neg=int(config_dataset['num_neg']),
                                    num_neg_test=int(config_dataset['num_neg_test'])
                                    )
    else:
        raise ValueError('unknown dataset name: ' + args.dataset)

def get_model(name, field_dims, config):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        return NeuralCollaborativeFiltering(field_dims, 
                                            embed_dim=int(config['embed_dim']), 
                                            mlp_dims=ast.literal_eval(config['mlp_dims']), 
                                            dropout=float(config['dropout_rate']))
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    
def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    total_batch = len(data_loader)
    for fields, target in data_loader:
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / total_batch

def test(model, data_loader, device):
    model.eval()
    metrics_k = np.zeros(4, dtype=float) # map, ndcg, precision, recall
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            predictions = model(fields)
            _, indices = torch.topk(predictions, args.top_k)
            recommends = torch.take(fields[:, [1]], indices).cpu().tolist()
            gt_item = fields[0][1].item() # leave one-out evaluation has only one item per user
            metrics_k += get_metrics_k(gt_item, recommends, len(data_loader))
            
    return metrics_k / len(data_loader)


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
        dataset = get_dataset()
        torch.save(dataset, f"{config_dataset['preprocessed_path']}/dataset.pt")
    else:
        dataset = torch.load(f"{config_dataset['preprocessed_path']}/dataset.pt")
        
    train_dataset = Subset(dataset, dataset.train_index)
    valid_dataset = Subset(dataset, dataset.valid_index)
    test_dataset = Subset(dataset, dataset.test_index)
    print(f"train length: {len(train_dataset)}, valid length: {len(valid_dataset)}, test length: {len(test_dataset)}")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=args.batch_size,
                                                    shuffle=True, 
                                                    num_workers=args.num_workers
                                                    )
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                    batch_size=int(config_dataset['num_neg_test']) + 1, 
                                                    shuffle=False, 
                                                    num_workers=args.num_workers
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=int(config_dataset['num_neg_test']) + 1,
                                                   shuffle=False, 
                                                   num_workers=args.num_workers
                                                   )
    
    model = get_model(args.model, dataset.field_dims, config_model).to(device)
    # print(f"model: \n {model}")
    # print("="*50)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=float(config_model['learning_rate']), 
                                 weight_decay=float(config_model['weight_decay'])
                                 )
    
    early_stopper = EarlyStopper(num_trials=2, direction='maximize' ,save_path=f'{args.save_dir}/{args.model}.pt')
    tk0 = tqdm(range(1, args.epochs+1), smoothing=0, mininterval=1.0)
    for epoch in tk0:
        loss = train(model, optimizer, train_data_loader, criterion, device)
        tk0.set_postfix(loss=loss)
        if epoch % 20 == 0:
            val_metrics = test(model, valid_data_loader, device)
            print(f"[Valid] mAP@K: {val_metrics[0]:.3f}, nDCG@K: {val_metrics[1]:.3f}, Precision@K:{val_metrics[2]:.3f}, Recall@K: {val_metrics[3]:.3f}")
            if not early_stopper.is_continuable(model, val_metrics[1]):
                print(f'[Valid] best nDCG@K: {early_stopper.best_metric:.3f}')
                break
    else: 
        print(f'[Valid] best nDCG@K: {early_stopper.best_metric:.3f}')
    
    tst_metrics = test(model, test_data_loader, device)        
    print(f"[Test] mAP@K: {tst_metrics[0]:.3f}, nDCG@K: {tst_metrics[1]:.3f}, Precision@K:{tst_metrics[2]:.3f}, Recall@K: {tst_metrics[3]:.3f}")  