import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import configparser
import ast

import torch
import tqdm
import numpy as np


from datasets.movielens import *
from models.FM import *
from models.NCF import *
from models.Wide_Deep import *
from models.NFM import *
from models.DeepFM import *
from models.xDeepFM import *

from evaluation import *
from utils import *

# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, 
        help="Random Seed.")
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--preprocessed_dataset', action='store_true', default=False,
        help=".")
    parser.add_argument('--model', default='ncf')
    parser.add_argument('--pretrained_model', action='store_true', default=False,
        help=".")
    parser.add_argument("--top_k", type=int, default=10, 
        help="Compute metrics@top_k.")
    parser.add_argument('--save_dir', default='../check_point/')
    parser.add_argument("--num_workers", type=int, default=4,
        help=".")
    parser.add_argument('--no_cuda', action='store_true', default=False,
        help='Disables CUDA training.')
    parser.add_argument("--gpu", type=str, default="0",
        help="GPU Card ID.")
    
    return parser.parse_args()

def get_data(name, preprocessed, config):
    if name == 'ml-1m':
        return MovieLens1M_Data(preprocessed, config)
    elif name == 'ml-20m':
        return MovieLens20M_Data(preprocessed, config)
    else:
        raise ValueError('unknown dataset name: ' + config['name'])

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
                                            dropout=float(config['dropout']),
                                            user_field_idx=np.array((0, ), dtype=np.compat.long),
                                            item_field_idx=np.array((1,), dtype=np.compat.long))
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
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    metrics_k = np.zeros(4, dtype=float)
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
    config = configparser.ConfigParser()
    config.read('../config/dataset.ini')
    config_dataset = config[args.dataset]
    config.read('../config/model.ini')
    config_model = config[args.model]
    
    device = torch.device("cuda:"+args.gpu if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"device: {device}, model: {args.model}, dataset: {args.dataset}, preprocessed_dataset: {args.preprocessed_dataset}")
        
    data = get_data(args.dataset, args.preprocessed_dataset, config_dataset)
    print(f"train length: {len(data.train_dataset)}, valid length: {len(data.valid_dataset)}, test length: {len(data.test_dataset)}")
        
    train_data_loader, valid_data_loader, test_data_loader = data.get_data_loaders(
                                                                int(config_model['batch_size']), 
                                                                args.num_workers * torch.cuda.device_count())
    
    model = get_model(args.model, data.field_dims, config_model).to(device)
    print(f"model: \n {model}")
    print("="*50)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=float(config_model['learning_rate']), weight_decay=float(config_model['weight_decay']))
    # early_stopper = EarlyStopper(num_trials=2, direction='minimize' ,save_path=f'{args.save_dir}/{args.model}.pt')
    
    # for epoch in range(int(config_model['epochs'])):
    for epoch in range(1):
        train(model, optimizer, train_data_loader, criterion, device)
        metrics = test(model, valid_data_loader, device)
        print(f"[Validation_{epoch+1}] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, Precision@K:{metrics[2]:.3f}, Recall@K: {metrics[3]:.3f}")
    #     if not early_stopper.is_continuable(model, loss):
    #         print(f'validation: best loss: {early_stopper.best_metric:.3f}')
    #         break
    # else: 
    #     print(f'validation: best loss: {early_stopper.best_metric}')
    
    # Test
    metrics = test(model, test_data_loader, device)
    print(f"[Test] mAP@K: {metrics[0]:.3f}, nDCG@K: {metrics[1]:.3f}, Precision@K:{metrics[2]:.3f}, Recall@K: {metrics[3]:.3f}")
    