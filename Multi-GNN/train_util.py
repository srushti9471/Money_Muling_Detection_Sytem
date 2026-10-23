import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import os
import numpy as np
from torch_geometric.sampler.base import HeteroSamplerOutput, SamplerOutput
import torch_geometric.sampler.neighbor_sampler as ns

def _py_hetero_neighbor_sample(self, seed_dict, seed_time_dict=None):
    if not hasattr(self, '_cached_np'):
        self._cached_np = {
            'colptr': {k: v.numpy() for k, v in self.colptr_dict.items()},
            'row': {k: v.numpy() for k, v in self.row_dict.items()}
        }
    colptr_dict = self._cached_np['colptr']
    row_dict = self._cached_np['row']
    
    node_dict = {}
    row_out = {}
    col_out = {}
    edge_out = {}
    
    all_nodes = {}
    frontier = {}
    for nt in self.node_types:
        seeds = seed_dict.get(nt, torch.empty(0, dtype=torch.long)).numpy()
        unique_seeds = np.unique(seeds).tolist()
        all_nodes[nt] = list(unique_seeds)
        frontier[nt] = unique_seeds
        
    sampled_edges = {et: [] for et in self.edge_types}
    sampled_src = {et: [] for et in self.edge_types}
    sampled_dst = {et: [] for et in self.edge_types}
    
    mapped_neighs = self.num_neighbors.get_mapped_values(self.edge_types)
    for hop in range(self.num_neighbors.num_hops):
        next_frontier = {nt: [] for nt in self.node_types}
        for et in self.edge_types:
            src_type, rel, dst_type = et
            k_key = '__'.join(et)
            colptr = colptr_dict[k_key]
            row = row_dict[k_key]
            k_neighs = int(mapped_neighs[k_key][hop])
            
            cur_frontier = np.array(frontier[dst_type], dtype=np.int64)
            valid_mask = cur_frontier < (len(colptr) - 1)
            valid_dsts = cur_frontier[valid_mask]
            if len(valid_dsts) == 0:
                continue
                
            starts = colptr[valid_dsts]
            ends = colptr[valid_dsts + 1]
            diffs = ends - starts
            has_edges = diffs > 0
            if not np.any(has_edges):
                continue
                
            starts = starts[has_edges]
            ends = ends[has_edges]
            dsts = valid_dsts[has_edges]
            
            for s, e, d in zip(starts, ends, dsts):
                deg = e - s
                if k_neighs < 0 or deg <= k_neighs:
                    chosen = np.arange(s, e)
                else:
                    chosen = np.random.choice(np.arange(s, e), size=k_neighs, replace=False)
                src_nodes = row[chosen]
                sampled_edges[et].extend(chosen)
                sampled_src[et].extend(src_nodes)
                sampled_dst[et].extend([d] * len(chosen))
                next_frontier[src_type].extend(src_nodes)
                
        for nt in self.node_types:
            if len(next_frontier[nt]) > 0:
                existing_set = set(all_nodes[nt])
                new_unique = [n for n in set(next_frontier[nt]) if n not in existing_set]
                all_nodes[nt].extend(new_unique)
                frontier[nt] = new_unique
            else:
                frontier[nt] = []
                
    node_to_local = {}
    for nt in self.node_types:
        node_arr = np.array(all_nodes[nt], dtype=np.int64)
        node_dict[nt] = torch.from_numpy(node_arr)
        node_to_local[nt] = {n: idx for idx, n in enumerate(all_nodes[nt])}
        
    for et in self.edge_types:
        src_type, rel, dst_type = et
        if len(sampled_edges[et]) > 0:
            edge_out[et] = torch.tensor(sampled_edges[et], dtype=torch.long)
            s_map = node_to_local[src_type]
            d_map = node_to_local[dst_type]
            r = [s_map[s] for s in sampled_src[et]]
            c = [d_map[d] for d in sampled_dst[et]]
            row_out[et] = torch.tensor(r, dtype=torch.long)
            col_out[et] = torch.tensor(c, dtype=torch.long)
        else:
            edge_out[et] = torch.empty(0, dtype=torch.long)
            row_out[et] = torch.empty(0, dtype=torch.long)
            col_out[et] = torch.empty(0, dtype=torch.long)
            
    return HeteroSamplerOutput(node=node_dict, row=row_out, col=col_out, edge=edge_out, batch=None)

def _py_homo_neighbor_sample(self, seed, seed_time=None):
    if not hasattr(self, '_cached_np_homo'):
        self._cached_np_homo = {
            'colptr': self.colptr.numpy(),
            'row': self.row.numpy()
        }
    colptr = self._cached_np_homo['colptr']
    row = self._cached_np_homo['row']
    
    seeds = seed.numpy()
    all_nodes = list(np.unique(seeds).tolist())
    frontier = list(all_nodes)
    
    sampled_edges = []
    sampled_src = []
    sampled_dst = []
    
    mapped_neighs = self.num_neighbors.get_mapped_values()
    for hop in range(self.num_neighbors.num_hops):
        next_frontier = []
        k_neighs = int(mapped_neighs[hop])
        cur_frontier = np.array(frontier, dtype=np.int64)
        valid_mask = cur_frontier < (len(colptr) - 1)
        valid_dsts = cur_frontier[valid_mask]
        if len(valid_dsts) == 0:
            continue
        starts = colptr[valid_dsts]
        ends = colptr[valid_dsts + 1]
        diffs = ends - starts
        has_edges = diffs > 0
        if not np.any(has_edges):
            continue
        starts = starts[has_edges]
        ends = ends[has_edges]
        dsts = valid_dsts[has_edges]
        for s, e, d in zip(starts, ends, dsts):
            deg = e - s
            if k_neighs < 0 or deg <= k_neighs:
                chosen = np.arange(s, e)
            else:
                chosen = np.random.choice(np.arange(s, e), size=k_neighs, replace=False)
            src_nodes = row[chosen]
            sampled_edges.extend(chosen)
            sampled_src.extend(src_nodes)
            sampled_dst.extend([d] * len(chosen))
            next_frontier.extend(src_nodes)
            
        if len(next_frontier) > 0:
            existing_set = set(all_nodes)
            new_unique = [n for n in set(next_frontier) if n not in existing_set]
            all_nodes.extend(new_unique)
            frontier = new_unique
        else:
            frontier = []
            
    node_arr = np.array(all_nodes, dtype=np.int64)
    node_tensor = torch.from_numpy(node_arr)
    node_to_local = {n: idx for idx, n in enumerate(all_nodes)}
    
    if len(sampled_edges) > 0:
        edge_tensor = torch.tensor(sampled_edges, dtype=torch.long)
        row_tensor = torch.tensor([node_to_local[s] for s in sampled_src], dtype=torch.long)
        col_tensor = torch.tensor([node_to_local[d] for d in sampled_dst], dtype=torch.long)
    else:
        edge_tensor = torch.empty(0, dtype=torch.long)
        row_tensor = torch.empty(0, dtype=torch.long)
        col_tensor = torch.empty(0, dtype=torch.long)
        
    return SamplerOutput(node=node_tensor, row=row_tensor, col=col_tensor, edge=edge_tensor, batch=None)

_orig_sample = ns.NeighborSampler._sample
def _safe_neighbor_sample(self, seed, seed_time=None):
    try:
        return _orig_sample(self, seed, seed_time)
    except ImportError:
        if isinstance(seed, dict):
            return _py_hetero_neighbor_sample(self, seed, seed_time)
        else:
            return _py_homo_neighbor_sample(self, seed, seed_time)

ns.NeighborSampler._sample = _safe_neighbor_sample


class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True, transform=transform)
        
        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
        
        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
    else:
        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform)
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        
    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args, precrec=False):
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    inds_cpu = inds.detach().cpu()
    all_edge_ids = loader.data.edge_attr.detach().cpu()[:, 0]
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        batch_edge_inds = inds_cpu[batch.input_id.detach().cpu()]
        batch_edge_ids = all_edge_ids[batch_edge_inds]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    if precrec:
        prec = precision_score(ground_truth, pred, zero_division=0)
        rec = recall_score(ground_truth, pred, zero_division=0)
        return f1, prec, rec

    return f1

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args, precrec=False):
    '''Evaluates the model performane for heterogenous graph data.'''
    preds = []
    ground_truths = []
    inds_cpu = inds.detach().cpu()
    all_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[:, 0]
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        batch_edge_inds = inds_cpu[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = all_edge_ids[batch_edge_inds]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch['node'].n_id
            add_edge_index = data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
            add_y = data['node', 'to', 'node'].y[missing_ids].detach().clone()
        
            batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
            batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
            batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    if precrec:
        prec = precision_score(ground_truth, pred, zero_division=0)
        rec = recall_score(ground_truth, pred, zero_division=0)
        return f1, prec, rec

    return f1

def save_model(model, optimizer, epoch, args, data_config):
    # Save the model in a dictionary
    save_dir = data_config["paths"]["model_to_save"]
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{save_dir}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar')
    
def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer