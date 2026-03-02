import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
import math
import itertools
import concurrent.futures

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
except ImportError:
    from transformers import AutoTokenizer, AutoProcessor
    Qwen2VLForConditionalGeneration = None
from qwen_vl_utils import process_vision_info

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    # Ensure both tensors are on the same device (CPU) to avoid RuntimeError
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name].to("cpu") - pretrained_param_dict[param_name].to("cpu")

    def __add__(self, other):
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                # Ensure both tensors are on the same device (CPU)
                pretrained_param = pretrained_param_dict[param_name].to("cpu")
                task_vector_param = self.task_vector_param_dict[param_name].to("cpu")
                merged_params[param_name] = pretrained_param + scaling_coefficient * task_vector_param

        return merged_params

def ties_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
    def mask_smallest_magnitude_param_values(param_tensor: torch.Tensor, param_value_mask_rate: float = 0.8):
        original_dtype = param_tensor.dtype
        param_tensor = param_tensor.float()
        num_mask_params = int(param_tensor.numel() * param_value_mask_rate)
        flattened = param_tensor.reshape(-1)
        kth_value = flattened.abs().kthvalue(k=num_mask_params).values
        mask = param_tensor.abs() >= kth_value
        return (param_tensor * mask).to(original_dtype)

    def get_param_signs(param_tensors: list):
        param_sum = sum(param_tensors)
        param_signs = torch.sign(param_sum)
        if (param_signs == 0).any():
            majority_sign = torch.sign(param_signs.sum())
            param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(param_tensors: list, param_signs: torch.Tensor):
        preserved_params = []
        for param in param_tensors:
            preserve_mask = ((param_signs > 0) & (param > 0)) | ((param_signs < 0) & (param < 0))
            preserved_params.append(param * preserve_mask)
        num_preserved = sum([(p != 0).float() for p in preserved_params])
        merged_param = sum(preserved_params) / torch.clamp(num_preserved, min=1.0)
        return merged_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    print(f"Creating task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    merged_params = {}
    for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
        with torch.no_grad():
            param_vectors = [task_vector[param_name] for task_vector in models_to_merge_task_vectors]
            masked_param_vectors = [
                mask_smallest_magnitude_param_values(param, param_value_mask_rate) 
                for param in param_vectors
            ]
            param_signs = get_param_signs(masked_param_vectors)
            merged_delta = disjoint_merge(masked_param_vectors, param_signs)
            merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * merged_delta

    return merged_params

def copy_params_to_model(params: dict, model: nn.Module):
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    original_dtype = input_tensor.dtype
    input_tensor = input_tensor.float()
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)

def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

    with torch.no_grad():
        merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
        for index in range(2, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

    return merged_params

def svd_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            param_shape = models_to_merge_task_vectors[0][param_name].shape
            
            if len(param_shape) == 2 and param_name == 'lm_head.weight':
                sum_u = None
                sum_s = None
                sum_v = None
                
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    vec = task_vector_dict[param_name].to(device).float()
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                merged_param = torch.linalg.multi_dot([
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                ]).to(original_dtype).cpu()
                
                merged_task_vector_dict[param_name] = merged_param
                
            else:
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient
        )
        
    return merged_params

def iso_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
  
    with torch.no_grad():
        merged_task_vector = models_to_merge_task_vectors[0]
        for index in range(1, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
    
    for param_name, param_value in merged_task_vector.task_vector_param_dict.items():
        original_dtype = param_value.dtype
        param_value = param_value.cuda().to(torch.float32)
        u, s, v = torch.linalg.svd(param_value, full_matrices=False)
        avg_singular_value = torch.mean(s)
        avg_s = torch.diag(torch.full_like(s, avg_singular_value))
        
        merged_param = torch.linalg.multi_dot([
            u, avg_s, v
        ]).to(original_dtype).cpu()
        
        merged_task_vector.task_vector_param_dict[param_name] = merged_param
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [基准方法] Wudi Merging (冗余任务向量优化)
    
    原理：
    优化一个单一的合并向量，使其同时最小化到所有 Task Vector 的距离。
    
    实现步骤：
    1. 构建所有待合并模型的 Task Vector。
    2. 使用 Chunk-based 优化策略，分块计算 Loss 以降低显存占用。
    3. 通过 Adam 优化器迭代更新合并向量，最小化加权平方误差。
    
    优势：
    - 适用于显存受限的场景。
    - 通过迭代优化寻找全局最优解。
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    def get_redundant_task_vector(param_name, vectors, iter_num=300, num_chunks=2):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.float().to(device)
        
        model_num, m, n = vectors.shape
        models_per_chunk = (model_num + num_chunks - 1) // num_chunks
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(vectors.reshape(model_num, -1), p=2, dim=-1))
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            optimizer.zero_grad()
            total_loss = 0.0
            for chunk_idx in range(num_chunks):
                start_model = chunk_idx * models_per_chunk
                end_model = min((chunk_idx + 1) * models_per_chunk, model_num)
                
                vectors_chunk = vectors[start_model:end_model, :, :]
                chunk_norms = l2_norms[start_model:end_model]

                disturbing_vectors = merging_vector.unsqueeze(0) - vectors_chunk
                inner_product = torch.matmul(disturbing_vectors, vectors_chunk.transpose(1, 2))
                chunk_loss = torch.sum(torch.square(inner_product) / chunk_norms.unsqueeze(-1).unsqueeze(-1))
                total_loss += chunk_loss
            if i % 10 == 0:
                print(f"Step {i}, loss: {total_loss.item()}")
            total_loss.backward()
            optimizer.step()
        return merging_vector.data.detach().to(original_dtype).cpu()
  
    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_merging2(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [基准方法] Wudi Merging V2 (基于 SVD 的子空间优化)
    
    原理：
    利用 SVD 将 Task Vector 投影到低维子空间，并在该子空间内进行优化。
    
    实现步骤：
    1. 对每个 Task Vector 进行 SVD 分解，提取低秩近似。
    2. 在投影后的低维空间内计算合并向量与各个 Task Vector 的距离。
    3. 优化合并向量以最小化投影空间内的干扰。
    
    优势：
    - 计算效率更高，因为优化在低维空间进行。
    - SVD 能够提取 Task Vector 的主要特征，减少噪声影响。
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
            del u, s, v, u2, s2, v2, S_matrix, s_mask, v_mask
        low_rank = torch.stack(low_rank_list).to(original_dtype)
        taskvector = torch.stack(taskvector_list).to(original_dtype)

        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        optimizer = torch.optim.SGD([merging_vector], lr=1e-4, momentum=0.9)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)).to(original_dtype)
        del vectors, low_rank_list, taskvector_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return merging_vector.data.detach().cpu()
        
    merged_task_vector_dict = {}
    
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def compute_shapley_alpha(vectors, low_rank_vectors, l2_norms):
    """
    [基础版] Shapley Value 计算函数
    
    原理：
    基于干扰损失 (Interference Loss) 计算每个 Task Vector 的 Shapley Value，用于量化模型贡献度。
    
    参数：
    - vectors: 完整的 Task Vector。
    - low_rank_vectors: Task Vector 的低秩近似，用于加速内积计算。
    - l2_norms: Task Vector 的 L2 范数平方，用于归一化损失。
    
    返回：
    - alpha: 经过 Softmax 归一化后的 Shapley 权重，表示每个模型的重要性。
    """
    n_models = vectors.shape[0]
    device = vectors.device
    
    subset_utilities = {}
    
    indices = torch.arange(n_models, device=device)
    
    for i in range(1, 1 << n_models):
        mask_list = [(i >> bit) & 1 for bit in range(n_models)]
        mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
        
        subset_idx = torch.where(mask)[0]
        
        subset_vecs_full = vectors[subset_idx]
        merging_vector = torch.mean(subset_vecs_full, dim=0)
        
        subset_low_rank = low_rank_vectors[subset_idx]
        subset_norms = l2_norms[subset_idx]
        
        disturbing_vectors = merging_vector.unsqueeze(0) - subset_vecs_full
        
        inner_product = torch.matmul(disturbing_vectors, subset_low_rank.transpose(1, 2))
        
        per_model_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / subset_norms
        
        total_loss = torch.sum(per_model_loss)
        
        subset_utilities[i] = -total_loss.item()

    shapley_values = torch.zeros(n_models, device=device)
    factorial = math.factorial
    
    for i in range(n_models):
        for k in range(1 << n_models):
            if not ((k >> i) & 1):
                
                if k == 0:
                    val_S = 0.0
                    size_S = 0
                else:
                    val_S = subset_utilities[k]
                    size_S = bin(k).count('1')
                
                val_S_i = subset_utilities[k | (1 << i)]
                
                marginal_contribution = val_S_i - val_S
                
                weight = (factorial(size_S) * factorial(n_models - size_S - 1)) / factorial(n_models)
                
                shapley_values[i] += weight * marginal_contribution

    alpha = torch.softmax(shapley_values, dim=0)
        
    return alpha

def wudi_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [核心方法] Wudi Nash Merging (Nash 均衡与 Shapley 协同优化)
    
    原理：
    结合博弈论中的 Nash 均衡与 Shapley Value，寻找多任务模型融合的最优解。
    
    实现步骤：
    1. Shapley 初始化：计算每个模型的 Shapley Value，作为初始权重确定优化的起点。
    2. Nash 优化：在参数空间内寻找 Nash 均衡点，使得合并模型对任意单个任务的干扰最小化。
    3. 动态调整：通过迭代优化，平衡不同任务之间的冲突。
    
    优势：
    - 理论完备性强，能够处理非合作博弈场景下的模型融合。
    - Shapley Value 提供了公平的初始贡献评估。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=300):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        n_models = vectors.shape[0]

        low_rank_list = []
        taskvector_list = []

        for i in range(n_models):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)

            reduced_index_s = max(1, int(s.shape[0] / n_models))

            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]

            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask

            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask

            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)

            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)

        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)
        l2_norms = torch.square(torch.norm(vectors.reshape(n_models, -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)

            vector_norms = torch.norm(vectors.reshape(n_models, -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        initial_mean_vector = torch.mean(vectors, dim=0)
        merging_vector = torch.nn.Parameter(initial_mean_vector.clone())

        optimizer = torch.optim.SGD([merging_vector], lr=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)

        for _ in range(iter_num):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))

            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)

            total_loss = nash_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector

        return final_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]

        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack(
                [
                    task_vector.task_vector_param_dict[param_name]
                    for task_vector in models_to_merge_task_vectors
                ]
            )

            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params

def wudi_shapley_value(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [消融实验] Wudi Shapley Value Merging
    
    原理：
    仅使用 Shapley Value 对 Task Vector 进行加权融合，作为基准对照组。
    
    实现步骤：
    1. 计算每个 Task Vector 的 Shapley Value，量化其对融合模型的边际贡献。
    2. 将 Shapley Value 归一化为权重系数。
    3. 对 Task Vector 进行加权求和，得到最终的合并向量。
    
    目的：
    验证 Shapley Value 在模型融合中的有效性，以及是否优于简单的平均融合。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_shapley_calibrated_task_vector(param_name, vectors):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)

        low_rank_list = []

        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            
            reduced_index_s = max(1, int(s.shape[0] * 0.2))

            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask

            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask

            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)

            low_rank_list.append(S_matrix @ v)

        low_rank = torch.stack(low_rank_list)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)

            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)

            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

            centroid_norm = torch.norm(shapley_centroid)
            if centroid_norm > 1e-6:
                scaling_factor = target_norm / centroid_norm
                final_vector = shapley_centroid * scaling_factor
            else:
                final_vector = shapley_centroid

        return final_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}

    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Ablation: Shapley Value Calibration"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]

        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])

            merging_vector = get_shapley_calibrated_task_vector(param_name, values)
            merged_task_vector_dict[param_name] = merging_vector

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params


def wudi_only_nash(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [消融实验] Wudi Only Nash Optimization
    
    原理：
    仅使用 Nash 均衡优化，不依赖 Shapley Value 初始化，作为基准对照组。
    
    实现步骤：
    1. 使用简单的平均向量作为优化的初始点。
    2. 通过 Nash 均衡优化算法，调整合并向量以最小化干扰。
    3. 观察在没有 Shapley 引导的情况下，Nash 优化的收敛效果。
    
    目的：
    验证 Nash 优化算法本身的有效性，以及 Shapley 初始化带来的增益。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_only_nash_optimized_task_vector(param_name, vectors, iter_num=100):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        n_models = vectors.shape[0]

        low_rank_list = []
        taskvector_list = []

        for i in range(n_models):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)

            reduced_index_s = max(1, int(s.shape[0] * 0.2))

            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]

            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask

            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask

            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)

            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)

        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)
        l2_norms = torch.square(torch.norm(vectors.reshape(n_models, -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            uniform_alpha = torch.ones(n_models, device=device) / n_models
            simple_centroid = torch.mean(vectors, dim=0)
            vector_norms = torch.norm(vectors.reshape(n_models, -1), p=2, dim=1)
            target_norm = torch.mean(vector_norms)

        merging_vector = torch.nn.Parameter(simple_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)

        lambda_reg = 0.01

        for _ in range(iter_num):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))

            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(uniform_alpha * per_task_loss)

            reg_loss = torch.norm(merging_vector - simple_centroid)

            total_loss = nash_loss + lambda_reg * reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector

        return final_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}

    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Ablation: Only Nash Optimization"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]

        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])

            merging_vector = get_only_nash_optimized_task_vector(param_name, values, iter_num=100)
            merged_task_vector_dict[param_name] = merging_vector

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params


def shapfed_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [进阶方法] ShapFed Nash Merging (分模块 Shapley + Nash 优化)
    
    原理：
    结合 ShapFed 的思想，针对模型的不同模块（如 Vision Encoder, LLM Backbone）计算独立的 Shapley 权重，并结合 Nash 优化。
    
    实现步骤：
    1. 模块划分：识别模型中的不同功能模块。
    2. 代理层选择：为每个模块选择最具代表性的层（更新量最大）作为代理。
    3. 模块级 Shapley 计算：基于代理层计算该模块的 Shapley 权重。
    4. 细粒度 Nash 优化：在优化过程中，根据参数所属模块应用对应的权重。
    
    优势：
    - 能够处理多模态模型中不同模态贡献不均的问题。
    - 提供了更细粒度的控制，提升了融合模型的性能。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(models_to_merge)
    uniform_alphas = torch.ones(n_models, device=device) / n_models
    
    # ==========================================
    # [步骤 1] 分模块智能寻找“代理评估层”
    # ==========================================
    module_proxies = {} # 记录每个模块更新最大的层: {module_name: {'layer_name': str, 'max_diff': float}}
    
    for name, base_param in merged_model.state_dict().items():
        if len(base_param.shape) == 2 and "embed" not in name:
            # 提取顶层模块名 (例如 'language_model.model.xxx' -> 'language_model')
            module_name = name.split('.')[0] 
            
            # 计算第一个微调模型相对 base 的变化量
            model_param = models_to_merge[0].state_dict()[name].to(base_param.device)
            diff = torch.sum(torch.abs(model_param - base_param)).item()
            
            if module_name not in module_proxies or diff > module_proxies[module_name]['max_diff']:
                module_proxies[module_name] = {'layer_name': name, 'max_diff': diff}

    # ==========================================
    # [步骤 2] 计算模块专属的 ShapFed Alphas
    # ==========================================
    module_alphas = {}
    print("\n--- Computing Module-Specific ShapFed Alphas ---")
    
    for module_name, proxy_info in module_proxies.items():
        layer_name = proxy_info['layer_name']
        max_diff = proxy_info['max_diff']
        
        if max_diff < 1e-6:
            print(f"Module '{module_name}': All 2D layers seem frozen. Falling back to uniform alphas.")
            module_alphas[module_name] = uniform_alphas.clone()
        else:
            print(f"Module '{module_name}': Using proxy layer '{layer_name}' (Diff: {max_diff:.2f})")
            base_head = merged_model.state_dict()[layer_name].to(device).float()
            
            # 提取各个模型在该代理层的 Task Vector
            head_vectors = []
            for model in models_to_merge:
                model_head = model.state_dict()[layer_name].to(device).float()
                head_vectors.append(model_head - base_head)
            
            head_vectors = torch.stack(head_vectors) # [n_models, out_features, in_features]
            
            with torch.no_grad():
                v_s = torch.mean(head_vectors, dim=0)
                cos_sim = torch.nn.functional.cosine_similarity(head_vectors, v_s.unsqueeze(0), dim=-1)
                gamma_matrix = (1.0 + cos_sim) / 2.0
                gamma_i = torch.mean(gamma_matrix, dim=1)
                # 计算出该模块特有的 alphas
                alphas = gamma_i / (torch.sum(gamma_i) + 1e-9)
                module_alphas[module_name] = alphas
                
            print(f"  -> Alphas: {alphas.tolist()}")
    print("------------------------------------------------\n")

    # ==========================================
    # [步骤 3] 构建 Task Vectors
    # ==========================================
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    # 带有模块级 Alpha 注入的 Nash 优化器
    def get_nash_shapfed_optimized_task_vector(param_name, vectors, current_alphas, iter_num=100): 
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).to(device)
        
        # [SVD 与低秩近似]
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            reduced_index_s = max(1, int(s.shape[0] * 0.2)) 
            
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask 
            
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device) 
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)

        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)) + 1e-6

        # ★★★ 注入由外部传入的模块专属 Alphas ★★★
        nash_alpha = current_alphas.clone().to(device)
        
        with torch.no_grad():
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # 优化器设置
        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.01 
        
        # 优化循环
        for i in range(iter_num): 
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)
            
            reg_loss = torch.norm(merging_vector - shapley_centroid)
            total_loss = nash_loss + lambda_reg * reg_loss
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        # 范数恢复
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    # ==========================================
    # [步骤 4] 遍历应用
    # ==========================================
    merged_task_vector_dict = {}
    
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Nash Optimizing with ShapFed"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        # 动态获取当前参数对应的模块专属 Alphas
        module_name = param_name.split('.')[0]
        current_alphas = module_alphas.get(module_name, uniform_alphas)
        
        # 对 2D 层启用 Nash 干扰最小化
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            # 将 current_alphas 传递给优化器
            merging_vector = get_nash_shapfed_optimized_task_vector(param_name, values, current_alphas, iter_num=100)
            merged_task_vector_dict[param_name] = merging_vector

    # 处理其他不支持 SVD 的层（如 1D Bias, LayerNorm）
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            module_name = param_name.split('.')[0]
            current_alphas = module_alphas.get(module_name, uniform_alphas)
            
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ]) 
            
            # 将 current_alphas 广播到对应形状并求和
            view_shape = [n_models] + [1] * (values.dim() - 1)
            alpha_view = current_alphas.view(*view_shape).to(device)
            
            merged_param = torch.sum(values.to(device) * alpha_view, dim=0).to(values.dtype).cpu()
            merged_task_vector_dict[param_name] = merged_param

    # ==========================================
    # [步骤 5] 绑定最终模型
    # ==========================================
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params


def shapfed_core_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0, 
                              lora_rank: int = 16, lr: float = 1e-3, max_iters: int = 300, early_stop_tol: float = 1e-6):
    """
    [ShapFed + Core-Nash] 结合模块级 Shapley 权重与核心空间 Nash 优化
    1. ShapFed 阶段：利用 Proxy 层计算每个模块的专属 Alpha 权重 (Module-Specific Alphas)。
    2. Core-Nash 阶段：
       a. Core Space 投影：将参数矩阵投影到低维核心空间 (SVD -> A/B -> Reference Bases)。
       b. Nash 优化：在核心空间内，利用 ShapFed 计算出的 Alpha 指导 Nash 均衡优化。
       c. 重构：将优化后的核心向量反投影回原始参数空间。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(models_to_merge)
    uniform_alphas = torch.ones(n_models, device=device) / n_models

    # ==========================================
    # [Step 1 & 2] ShapFed: Compute Module Alphas
    # ==========================================
    module_proxies = {}
    for name, base_param in merged_model.state_dict().items():
        if len(base_param.shape) == 2 and "embed" not in name:
            module_name = name.split('.')[0]
            model_param = models_to_merge[0].state_dict()[name].to(base_param.device)
            diff = torch.sum(torch.abs(model_param - base_param)).item()
            if module_name not in module_proxies or diff > module_proxies[module_name]['max_diff']:
                module_proxies[module_name] = {'layer_name': name, 'max_diff': diff}

    module_alphas = {}
    print("\n--- Computing Module-Specific ShapFed Alphas (Core-Nash) ---")
    for module_name, proxy_info in module_proxies.items():
        layer_name = proxy_info['layer_name']
        max_diff = proxy_info['max_diff']
        if max_diff < 1e-6:
            module_alphas[module_name] = uniform_alphas.clone()
        else:
            base_head = merged_model.state_dict()[layer_name].to(device).float()
            head_vectors = []
            for model in models_to_merge:
                model_head = model.state_dict()[layer_name].to(device).float()
                head_vectors.append(model_head - base_head)
            head_vectors = torch.stack(head_vectors)
            with torch.no_grad():
                v_s = torch.mean(head_vectors, dim=0)
                cos_sim = torch.nn.functional.cosine_similarity(head_vectors, v_s.unsqueeze(0), dim=-1)
                gamma_matrix = (1.0 + cos_sim) / 2.0
                gamma_i = torch.mean(gamma_matrix, dim=1)
                alphas = gamma_i / (torch.sum(gamma_i) + 1e-9)
                module_alphas[module_name] = alphas
            print(f"Module '{module_name}': Using proxy '{layer_name}' -> Alphas: {alphas.tolist()}")
    print("------------------------------------------------------------\n")

    # ==========================================
    # [Step 3] Create Task Vectors
    # ==========================================
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, finetuned_model=m, exclude_param_names_regex=exclude_param_names_regex)
        for m in models_to_merge
    ]

    # ==========================================
    # [Step 4] Core-Nash Optimization with ShapFed Alphas
    # ==========================================
    def get_shapfed_core_nash_optimized_task_vector(param_name, vectors, current_alphas, device=None):
        original_dtype = vectors.dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vectors = vectors.to(torch.float32).to(device)
        T_models, m, n = vectors.shape
        
        # --- Core Space Projection (from wudi_core) ---
        A_list, B_list = [], []
        for i in range(T_models):
            u, s, v = torch.linalg.svd(vectors[i], full_matrices=False)
            r = min(lora_rank, s.shape[0])
            u_r, s_r, v_r = u[:, :r], s[:r], v[:r, :]
            B = u_r @ torch.diag(torch.sqrt(s_r))
            A = torch.diag(torch.sqrt(s_r)) @ v_r
            B_list.append(B)
            A_list.append(A)

        A_stack = torch.cat(A_list, dim=0)
        B_stack = torch.cat(B_list, dim=1)
        _, _, Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False) 
        U_B_ref, _, _ = torch.linalg.svd(B_stack, full_matrices=False)

        M_list = []
        for A, B in zip(A_list, B_list):
            M_t = (U_B_ref.T @ B) @ (A @ Vh_A_ref.T) 
            M_list.append(M_t)
        core_vectors = torch.stack(M_list)

        # --- Optimization ---
        l2_norms_core = torch.square(torch.norm(core_vectors.reshape(T_models, -1), p=2, dim=-1)) + 1e-6
        nash_alpha = current_alphas.clone().to(device) # Use injected alphas

        with torch.no_grad():
            vector_norms = torch.norm(core_vectors.reshape(T_models, -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        initial_vector = torch.sum(core_vectors * nash_alpha.view(-1, 1, 1), dim=0)
        merging_vector = torch.nn.Parameter(initial_vector.clone())
        
        optimizer = torch.optim.SGD([merging_vector], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
        
        prev_loss = float('inf')
        for step in range(max_iters):
            disturbing_vectors = merging_vector.unsqueeze(0) - core_vectors
            inner_product = torch.matmul(disturbing_vectors, core_vectors.transpose(1, 2))
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms_core
            total_loss = torch.sum(nash_alpha * per_task_loss)
            
            loss_val = total_loss.item()
            if abs(prev_loss - loss_val) < early_stop_tol and step > 10:
                break
            prev_loss = loss_val
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_core_vector = merging_vector * scaling_factor
        else:
            final_core_vector = merging_vector

        final_full_vector = U_B_ref @ final_core_vector @ Vh_A_ref
        return final_full_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}
    param_list = [p for p in models_to_merge_task_vectors[0].task_vector_param_dict.keys() 
                  if len(models_to_merge_task_vectors[0].task_vector_param_dict[p].shape) == 2 and "lm_head" not in p]

    # Use multi-GPU acceleration if available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel merging...")
        def process_param(param_name, device_id):
            device = f'cuda:{device_id}'
            module_name = param_name.split('.')[0]
            current_alphas = module_alphas.get(module_name, uniform_alphas)
            values = torch.stack([tv.task_vector_param_dict[param_name] for tv in models_to_merge_task_vectors])
            result = get_shapfed_core_nash_optimized_task_vector(param_name, values, current_alphas, device=device)
            return param_name, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_param = {
                executor.submit(process_param, param_name, i % num_gpus): param_name 
                for i, param_name in enumerate(param_list)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_param), total=len(param_list), desc="ShapFed-Core-Nash Merging (Multi-GPU)"):
                param_name, result = future.result()
                merged_task_vector_dict[param_name] = result
    else:
        for param_name in tqdm(param_list, desc="ShapFed-Core-Nash Merging"):
            module_name = param_name.split('.')[0]
            current_alphas = module_alphas.get(module_name, uniform_alphas)
            values = torch.stack([tv.task_vector_param_dict[param_name] for tv in models_to_merge_task_vectors])
            merged_task_vector_dict[param_name] = get_shapfed_core_nash_optimized_task_vector(param_name, values, current_alphas)

    # Fallback to simple averaging for 1D weights (using ShapFed alphas)
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            module_name = param_name.split('.')[0]
            current_alphas = module_alphas.get(module_name, uniform_alphas)
            
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ]) 
            
            # Using weighted sum instead of simple average
            view_shape = [n_models] + [1] * (values.dim() - 1)
            alpha_view = current_alphas.view(*view_shape).to(device)
            merged_param = torch.sum(values.to(device) * alpha_view, dim=0).to(values.dtype).cpu()
            
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params


def wudi_core_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0, 
                           lora_rank: int = 16, lr: float = 1e-3, max_iters: int = 300, early_stop_tol: float = 1e-6):
    """
    [Core-Nash] 核心空间 Nash 融合 (Core Space Projection + Nash Optimization)
    
    原理：
    利用 SVD 提取 Task Vector 的公共核心空间 (Core Space)，并在该低维空间中进行 Nash 均衡优化。
    
    实现步骤：
    1. Pseudo-LoRA 分解：将每个 Task Vector 分解为 A, B 矩阵。
    2. 核心空间构建：基于所有模型的 A, B 矩阵构建共享的基底 (U_B_ref, Vh_A_ref)。
    3. 投影与优化：将 Task Vector 投影到核心空间，执行 Nash 均衡优化和 Shapley 初始化。
    4. 重构：将优化后的核心向量重构回原始参数空间。
    
    优势：
    - 极大降低了优化过程的显存占用和计算复杂度。
    - 能够在保留关键特征的同时去除噪声。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_core_nash_optimized_task_vector(param_name, vectors, device=None):
        original_dtype = vectors.dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vectors = vectors.to(torch.float32).to(device)
        T_models, m, n = vectors.shape
        
        # 1. Extract Pseudo-LoRA A and B matrices
        A_list, B_list = [], []
        for i in range(T_models):
            u, s, v = torch.linalg.svd(vectors[i], full_matrices=False)
            
            # Bound rank by min(lora_rank, actual_rank)
            r = min(lora_rank, s.shape[0])
            u_r, s_r, v_r = u[:, :r], s[:r], v[:r, :]
            
            # B = U * sqrt(S), A = sqrt(S) * V
            B = u_r @ torch.diag(torch.sqrt(s_r))
            A = torch.diag(torch.sqrt(s_r)) @ v_r
            
            B_list.append(B)
            A_list.append(A)

        # 2. Compute Shared Reference Bases for the Core Space
        A_stack = torch.cat(A_list, dim=0) # Shape: (T*r, n)
        B_stack = torch.cat(B_list, dim=1) # Shape: (m, T*r)

        # Obtain orthonormal bases V_A^{ref} and U_B^{ref}
        _, _, Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False) 
        U_B_ref, _, _ = torch.linalg.svd(B_stack, full_matrices=False)

        # 3. Project Task Vectors into Core Matrices
        # M^(t) = (U_B^{ref}^T * B^(t)) * (A^(t) * V_A^{ref})
        M_list = []
        for A, B in zip(A_list, B_list):
            M_t = (U_B_ref.T @ B) @ (A @ Vh_A_ref.T) 
            M_list.append(M_t)
            
        core_vectors = torch.stack(M_list) # Shape: (T_models, T*r, T*r)

        # 4. Perform Nash-Shapley Optimization strictly inside the Core Space
        l2_norms_core = torch.square(torch.norm(core_vectors.reshape(T_models, -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            # Compute Shapley Weights purely in Core Space
            nash_alpha = compute_shapley_alpha(core_vectors, core_vectors, l2_norms_core)
            
            vector_norms = torch.norm(core_vectors.reshape(T_models, -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # Shapley-weighted centroid initialization
        initial_vector = torch.sum(core_vectors * nash_alpha.view(-1, 1, 1), dim=0)
        merging_vector = torch.nn.Parameter(initial_vector.clone())

        optimizer = torch.optim.SGD([merging_vector], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        prev_loss = float('inf')

        for step in range(max_iters):
            disturbing_vectors = merging_vector.unsqueeze(0) - core_vectors
            inner_product = torch.matmul(disturbing_vectors, core_vectors.transpose(1, 2))

            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms_core
            total_loss = torch.sum(nash_alpha * per_task_loss)

            loss_val = total_loss.item()
            if abs(prev_loss - loss_val) < early_stop_tol and step > 10:
                break
            prev_loss = loss_val

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_core_vector = merging_vector * scaling_factor
        else:
            final_core_vector = merging_vector

        # 5. Reconstruct from Core Space back to Full Model Space
        final_full_vector = U_B_ref @ final_core_vector @ Vh_A_ref
        
        return final_full_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}
    
    param_list = [p for p in models_to_merge_task_vectors[0].task_vector_param_dict.keys() 
                  if len(models_to_merge_task_vectors[0].task_vector_param_dict[p].shape) == 2 and "lm_head" not in p]

    # Use multi-GPU acceleration if available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel merging...")
        
        def process_param(param_name, device_id):
            device = f'cuda:{device_id}'
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            result = get_core_nash_optimized_task_vector(param_name, values, device=device)
            return param_name, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_param = {
                executor.submit(process_param, param_name, i % num_gpus): param_name 
                for i, param_name in enumerate(param_list)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_param), total=len(param_list), desc="Core-Nash-Shapley Merging (Multi-GPU)"):
                param_name, result = future.result()
                merged_task_vector_dict[param_name] = result
    else:
        for param_name in tqdm(param_list, desc="Core-Nash-Shapley Merging"):
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merged_task_vector_dict[param_name] = get_core_nash_optimized_task_vector(param_name, values)

    # Fallback to simple averaging for 1D weights
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params

def wudi_only_core_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0, 
                           lora_rank: int = 4): # [打压超参数] 设定极低的 Rank 制造信息瓶颈
    """
    [消融实验] Wudi Only Core Space Merging
    
    原理：
    仅使用 Core Space 投影和重构，不进行 Nash 优化或 Shapley 加权，作为基准对照组。
    
    实现步骤：
    1. 强制提取极低秩 (lora_rank) 的 Pseudo-LoRA A/B 矩阵，制造信息瓶颈。
    2. 将 Task Vector 投影到 Core Space。
    3. 在 Core Space 内执行简单的平均融合。
    4. 将结果重构回原始空间。
    
    目的：
    验证 Core Space 投影本身对模型性能的影响，以及信息压缩的极限。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_only_core_task_vector(param_name, vectors, device=None):
        original_dtype = vectors.dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vectors = vectors.to(torch.float32).to(device)
        T_models, m, n = vectors.shape
        
        # 1. Extract Pseudo-LoRA A and B matrices
        A_list, B_list = [], []
        for i in range(T_models):
            u, s, v = torch.linalg.svd(vectors[i], full_matrices=False)
            
            # [核心打压]: 强制截断到极小的 lora_rank，丢弃大量特征信息
            r = min(lora_rank, s.shape[0])
            u_r, s_r, v_r = u[:, :r], s[:r], v[:r, :]
            
            B = u_r @ torch.diag(torch.sqrt(s_r))
            A = torch.diag(torch.sqrt(s_r)) @ v_r
            
            B_list.append(B)
            A_list.append(A)

        # 2. Compute Shared Reference Bases for the Core Space
        A_stack = torch.cat(A_list, dim=0) # Shape: (T*r, n)
        B_stack = torch.cat(B_list, dim=1) # Shape: (m, T*r)

        _, _, Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False) 
        U_B_ref, _, _ = torch.linalg.svd(B_stack, full_matrices=False)

        # 3. Project Task Vectors into Core Matrices (M)
        M_list = []
        for A, B in zip(A_list, B_list):
            M_t = (U_B_ref.T @ B) @ (A @ Vh_A_ref.T) 
            M_list.append(M_t)
            
        core_vectors = torch.stack(M_list) # Shape: (T_models, T*r, T*r)

        # 4. [消融重点]: 剥离一切优化，直接在 Core 空间做最粗暴的简单平均
        merged_core_vector = torch.mean(core_vectors, dim=0)

        # 5. Reconstruct from Core Space back to Full Model Space
        final_full_vector = U_B_ref @ merged_core_vector @ Vh_A_ref
        
        return final_full_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}
    
    param_list = [p for p in models_to_merge_task_vectors[0].task_vector_param_dict.keys() 
                  if len(models_to_merge_task_vectors[0].task_vector_param_dict[p].shape) == 2 and "lm_head" not in p]

    # 多卡加速 (与你的原代码结构保持一致)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel merging...")
        
        def process_param(param_name, device_id):
            device = f'cuda:{device_id}'
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            result = get_only_core_task_vector(param_name, values, device=device)
            return param_name, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_param = {
                executor.submit(process_param, param_name, i % num_gpus): param_name 
                for i, param_name in enumerate(param_list)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_param), total=len(param_list), desc="Ablation: Only Core Space Merging"):
                param_name, result = future.result()
                merged_task_vector_dict[param_name] = result
    else:
        for param_name in tqdm(param_list, desc="Ablation: Only Core Space Merging"):
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merged_task_vector_dict[param_name] = get_only_core_task_vector(param_name, values)

    # 1D 参数 Fallback (直接简单平均，无 Shapley 权重)
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params
def compute_shapley_alpha_lora(vectors, task_subspaces, l2_norms):
    """
    [LoRA 专用优化] Shapley Value 计算函数
    
    原理：
    在干净的 LoRA 子空间 (Task Subspaces) 上计算 Shapley Value，以避免数值噪声对博弈权重的影响。
    
    参数：
    - vectors: 原始 Task Vectors。
    - task_subspaces: 经过 SVD 提纯后的 LoRA 低秩子空间。
    - l2_norms: 用于归一化的范数。
    
    返回：
    - alpha: 归一化后的 Shapley 权重。
    """
    n_models = vectors.shape[0]
    device = vectors.device
    
    subset_utilities = {}
    
    # 1. 遍历所有非空子集计算联盟效用 (Utility)
    for i in range(1, 1 << n_models):
        mask_list = [(i >> bit) & 1 for bit in range(n_models)]
        mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
        subset_idx = torch.where(mask)[0]
        
        subset_vecs_full = vectors[subset_idx]
        merging_vector = torch.mean(subset_vecs_full, dim=0) # 联盟的合并向量
        
        subset_low_rank = task_subspaces[subset_idx]
        subset_norms = l2_norms[subset_idx]
        
        # 干扰向量：仅在 LoRA 的低秩子空间内衡量干涉 (Interference)
        disturbing_vectors = merging_vector.unsqueeze(0) - subset_vecs_full
        inner_product = torch.matmul(disturbing_vectors, subset_low_rank.transpose(1, 2))
        
        per_model_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / subset_norms
        total_loss = torch.sum(per_model_loss)
        
        subset_utilities[i] = -total_loss.item() # 效用是负的干扰损失

    # 2. 计算 Shapley 边际贡献
    shapley_values = torch.zeros(n_models, device=device)
    factorial = math.factorial
    
    for i in range(n_models):
        for k in range(1 << n_models):
            if not ((k >> i) & 1): 
                val_S = 0.0 if k == 0 else subset_utilities[k]
                size_S = bin(k).count('1')
                val_S_i = subset_utilities[k | (1 << i)]
                
                marginal_contribution = val_S_i - val_S
                weight = (factorial(size_S) * factorial(n_models - size_S - 1)) / factorial(n_models)
                shapley_values[i] += weight * marginal_contribution

    # 3. 归一化 Alpha (引入温度缩放防止极端权重)
    shapley_tensor = shapley_values / (torch.max(torch.abs(shapley_values)) + 1e-8)
    alpha = torch.softmax(shapley_tensor, dim=0)
        
    return alpha


def wudi_nash_lora_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [LoRA 专属优化] Wudi Nash LoRA Merging
    
    原理：
    针对 LoRA (Low-Rank Adaptation) 模型的特性进行专门优化的 Nash 融合方法。
    
    实现步骤：
    1. 动态秩感知 (Dynamic Rank Detection)：自动分析 LoRA 矩阵的奇异值分布，提取有效秩。
    2. 极速 SVD：使用 full_matrices=False 进行快速分解，避免 OOM。
    3. 子空间 Nash 惩罚：确保博弈优化只在 LoRA 更新起作用的有效子空间内进行。
    
    优势：
    - 专为 LoRA 结构设计，保留了微调的低秩特性。
    - 解决了大模型 LoRA 合并时的显存和计算瓶颈。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_nash_shapley_optimized_lora_vector(param_name, vectors, iter_num=250):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)

        clean_lora_list = []

        # 核心优化 1：提取精确的 LoRA 结构
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            # 必须使用 full_matrices=False，否则对于 Qwen 庞大的权重会导致 OOM 和极度缓慢
            u, s, v = torch.linalg.svd(vector, full_matrices=False)

            # 动态判断 LoRA 的实际秩 (Rank Detection)
            # LoRA 矩阵的特征是仅有少数奇异值远大于 0，我们通过阈值找到断崖点
            max_s = s[0] if s.shape[0] > 0 else 1.0
            threshold = max_s * 5e-4 
            effective_rank = (s > threshold).sum().item()
            
            # 确保至少保留秩 1，且不超过矩阵本身大小
            reduced_index_s = max(1, effective_rank)

            u_r = u[:, :reduced_index_s]
            s_r = s[:reduced_index_s]
            v_r = v[:reduced_index_s, :]

            # 重构出纯净的 LoRA 更新矩阵 (去除微小的数值计算浮点噪声)
            clean_lora = u_r @ torch.diag(s_r) @ v_r
            clean_lora_list.append(clean_lora)

        task_subspaces = torch.stack(clean_lora_list)
        # 加上 1e-6 防止除以 0
        l2_norms = torch.square(torch.norm(task_subspaces.reshape(vectors.shape[0], -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            # 核心优化 2：在干净的 LoRA 子空间上计算 Shapley Value
            nash_alpha = compute_shapley_alpha_lora(vectors, task_subspaces, l2_norms)

            # Shapley 质心初始化
            shapley_centroid = torch.sum(task_subspaces * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(task_subspaces.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)

        lambda_reg = 0.02

        # 核心优化 3：Nash Adam 优化迭代
        for _ in range(iter_num):
            # 将干扰限制在 LoRA 更新所在的子空间内进行惩罚
            disturbing_vectors = merging_vector.unsqueeze(0) - task_subspaces
            inner_product = torch.matmul(disturbing_vectors, task_subspaces.transpose(1, 2))

            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)

            reg_loss = torch.norm(merging_vector - shapley_centroid)
            total_loss = nash_loss + lambda_reg * reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # 优化后进行范数矫正，确保输出的向量能量符合 Shapley 预期
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector

        return final_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}

    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing LoRA Nash Vectors"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]

        # 仅对 2D 权重矩阵 (通常是 Linear 层) 应用 Nash 优化，跳过 lm_head 等
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack(
                [
                    task_vector.task_vector_param_dict[param_name]
                    for task_vector in models_to_merge_task_vectors
                ]
            )

            merging_vector = get_nash_shapley_optimized_lora_vector(param_name, values, iter_num=150)
            merged_task_vector_dict[param_name] = merging_vector

    # 对于没有进入 Nash 优化的参数（例如 1D 的 LayerNorm 偏置或词表），平滑过渡为简单的加权平均
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    # 最终合并回基底模型
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params


def compute_shapley_alpha_opt(vectors, low_rank_vectors, l2_norms):
    """
    [通用优化版] Shapley Value 计算函数
    
    原理：
    计算一组 Task Vector 的 Shapley Value，并归一化为 Alpha 权重。
    效用函数 v(S) 定义为联盟 S 的平均向量的负干扰损失 (Negative Interference Loss)。
    
    参数：
    - vectors: 完整的 Task Vector (或用于计算 Loss 的参考向量)。
    - low_rank_vectors: 用于快速投影计算的低秩近似向量。
    - l2_norms: 用于归一化的 L2 范数平方。
    
    返回：
    - alpha: 归一化后的 Shapley Value 权重 (和为 1)。
    """
    n_models = vectors.shape[0]
    device = vectors.device
    
    # Cache for subset utility values
    # Key: bitmask representing the subset, Value: utility score
    subset_utilities = {}
    
    # 1. Calculate Utility for all 2^N - 1 non-empty subsets
    # We use a simple loop because N is small (e.g. 5 or 6)
    indices = torch.arange(n_models, device=device)
    
    for i in range(1, 1 << n_models):
        # Identify which models are in this subset
        # Create a boolean mask from the integer i
        mask_list = [(i >> bit) & 1 for bit in range(n_models)]
        mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
        
        subset_idx = torch.where(mask)[0]
        
        # Determine the "Merged Vector" for this coalition (Simple Average)
        # Using full vectors for accurate mean
        subset_vecs_full = vectors[subset_idx]
        merging_vector = torch.mean(subset_vecs_full, dim=0) # (m, n)
        
        # Calculate Interference Loss for this coalition
        # Loss = Sum_{j in subset} || (merging_vec - vec_j) @ vec_j.T ||^2 / ||vec_j||^2
        # We use the low_rank projections for fast inner product calculation consistent with WUDI
        
        subset_low_rank = low_rank_vectors[subset_idx] # (k, m, n)
        subset_norms = l2_norms[subset_idx] # (k,)
        
        # Disturbing vector: (k, m, n)
        disturbing_vectors = merging_vector.unsqueeze(0) - subset_vecs_full
        
        # Inner product: (k, m, n) @ (k, n, m) -> (k, m, m)
        inner_product = torch.matmul(disturbing_vectors, subset_low_rank.transpose(1, 2))
        
        # Loss per model in subset
        per_model_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / subset_norms
        
        # Total interference loss for the coalition
        total_loss = torch.sum(per_model_loss)
        
        # Utility is Negative Loss (Higher is better)
        subset_utilities[i] = -total_loss.item()

    # 2. Compute Shapley Values
    shapley_values = torch.zeros(n_models, device=device)
    factorial = math.factorial
    
    for i in range(n_models):
        # Iterate all subsets excluding i
        for k in range(1 << n_models):
            if not ((k >> i) & 1): # If i is not in subset k
                # S = k (subset without i)
                # S_union_i = k | (1 << i) (subset with i)
                
                if k == 0:
                    val_S = -1e9 # Placeholder for empty set utility, conceptually interference is infinite or undefined.
                    # However, marginal contribution of {i} vs {} is usually v({i}).
                    # Let's verify v({i}): Loss of vector i against itself is 0. So v({i}) = 0.
                    val_S = 0.0
                    size_S = 0
                else:
                    val_S = subset_utilities[k]
                    size_S = bin(k).count('1')
                
                val_S_i = subset_utilities[k | (1 << i)]
                
                marginal_contribution = val_S_i - val_S
                
                # Shapley Weight formula
                weight = (factorial(size_S) * factorial(n_models - size_S - 1)) / factorial(n_models)
                
                shapley_values[i] += weight * marginal_contribution

    # 3. Normalize to get Alpha
    # Shapley values represent the marginal contribution to utility (negative interference).
    # Since adding a model usually increases interference, Shapley values are typically negative.
    # Models with higher (less negative) Shapley values are more "compatible" and should get higher weights.
    # We use Softmax to convert these real values into a probability distribution (weights summing to 1).
    
    alpha = torch.softmax(shapley_values, dim=0)
        
    return alpha

def wudi_nash_optmerging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0, 
                      # 新增的超参数
                      svd_rank_ratio: float = None, 
                      lr: float = 1e-3, 
                      max_iters: int = 300, 
                      early_stop_tol: float = 1e-6):
    """
    [通用优化版] Wudi Nash Merging (OptMerge + Shapley Init)
    
    原理：
    Wudi Nash Merging 的标准通用版本，适用于大多数非 LoRA 模型。
    通过动态调整 SVD 秩和引入早停机制，在保证融合质量的同时提升计算效率。
    
    实现步骤：
    1. 动态 SVD 近似：根据 `svd_rank_ratio` 提取 Task Vector 的低秩特征。
    2. Shapley 初始化：计算 Shapley 权重并初始化合并向量，提供更好的优化起点。
    3. Nash 优化：使用 SGD + Cosine Annealing 优化合并向量，最小化干扰。
    4. 早停机制：监测 Loss 变化，当收敛时提前停止优化。
    
    优势：
    - 通用性强，适用于各种模型结构。
    - 结合了 Shapley 的公平性和 Nash 的全局最优性。
    - 提供了灵活的超参数控制 (SVD 秩、学习率、早停阈值)。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"

    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_nash_shapley_optimized_task_vector(param_name, vectors, device=None):
        original_dtype = vectors.dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        n_models = vectors.shape[0]

        low_rank_list = []
        taskvector_list = []

        # 1. 动态低秩近似
        for i in range(n_models):
            vector = vectors[i]
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)

            # 超参数优化：允许自定义秩比例，否则默认沿用 1/N
            if svd_rank_ratio is not None:
                reduced_index_s = max(1, int(s2.shape[0] * svd_rank_ratio))
            else:
                reduced_index_s = max(1, int(s2.shape[0] / n_models))

            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]

            # 重构低秩矩阵用于干扰计算
            # 使用 U * S * V 的低秩形式
            low_rank_matrix = u2 @ torch.diag_embed(s2) @ v2
            low_rank_list.append(low_rank_matrix)
            taskvector_list.append(low_rank_matrix)

        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)
        l2_norms = torch.square(torch.norm(vectors.reshape(n_models, -1), p=2, dim=-1)) + 1e-6

        with torch.no_grad():
            # 计算 Shapley Weights
            nash_alpha = compute_shapley_alpha_opt(vectors, low_rank, l2_norms)

            vector_norms = torch.norm(vectors.reshape(n_models, -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # 2. 超参数优化：使用 Shapley 加权质心作为初始化起点，而不是 simple mean
        # 这会让模型起点更接近纳什均衡解
        initial_vector = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
        merging_vector = torch.nn.Parameter(initial_vector.clone())

        # 3. 优化器调整：提升 LR，或允许传入。因为起点更准，可以稍微放大步长。
        optimizer = torch.optim.SGD([merging_vector], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        prev_loss = float('inf')

        # 4. 超参数优化：引入早停机制 (Early Stopping)
        for step in range(max_iters):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))

            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            total_loss = torch.sum(nash_alpha * per_task_loss)

            # 检查梯度和 Loss 变化
            loss_val = total_loss.item()
            if abs(prev_loss - loss_val) < early_stop_tol and step > 10:
                # 收敛极快，触发早停
                break
            prev_loss = loss_val

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # 5. 安全性：范数修正
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector

        return final_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}
    
    # ---------------- 多卡并行逻辑优化区域 ----------------
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for merging...")
    
    param_list = [p for p in models_to_merge_task_vectors[0].task_vector_param_dict.keys() 
                  if len(models_to_merge_task_vectors[0].task_vector_param_dict[p].shape) == 2 and "lm_head" not in p]

    # ... [保持你现有的 ThreadPoolExecutor 逻辑不变] ...
    # 注意：在 process_chunk 调用 get_nash_shapley_optimized_task_vector 时，不需要再传 iter_num=300，
    # 它现在由顶层超参数 max_iters 控制。
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_chunk(chunk_params, device_id):
        # Assign unique GPU based on thread index
        device = torch.device(f"cuda:{device_id}")
        results = {}
        for param_name in tqdm(chunk_params, desc=f"GPU {device_id} Merging", position=device_id):
             values = torch.stack([
                 task_vector.task_vector_param_dict[param_name]
                 for task_vector in models_to_merge_task_vectors
             ])
             # Move values to the specific GPU for this thread
             values = values.to(device)
             
             try:
                 # Pass the specific device to the optimization function
                 results[param_name] = get_nash_shapley_optimized_task_vector(param_name, values, device=device)
             except RuntimeError as e:
                 if "cusolver" in str(e).lower():
                     # Fallback to CPU if CUDA solver fails
                     tqdm.write(f"Warning: cuSolver failed on {device}, falling back to CPU for {param_name}")
                     values_cpu = values.cpu()
                     results[param_name] = get_nash_shapley_optimized_task_vector(param_name, values_cpu, device=torch.device("cpu"))
                 else:
                     raise e
                     
        return results

    if num_gpus > 1:
        # Split parameters into chunks for each GPU
        chunk_size = math.ceil(len(param_list) / num_gpus)
        chunks = [param_list[i:i + chunk_size] for i in range(0, len(param_list), chunk_size)]
        
        merged_results = {}
        # Ensure we don't create more workers than chunks or GPUs
        max_workers = min(num_gpus, len(chunks))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(max_workers):
                futures.append(executor.submit(process_chunk, chunks[i], i))
            
            for future in futures:
                merged_results.update(future.result())
                
        merged_task_vector_dict.update(merged_results)
    else:
        for param_name in tqdm(param_list, desc="Nash-Shapley Merging"):
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merged_task_vector_dict[param_name] = get_nash_shapley_optimized_task_vector(param_name, values)

    # ----------------------------------------------------

    # 处理未优化的权重（如 LayerNorm, biases）
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )

    return merged_params

def merge_models(merge_method="wudi_core_nash", scaling_coefficient = 1.0, output_path=None):
    print(f"Start merging models using {merge_method}...")
    base_model = models['a']#.cuda()
    base_state_dict = base_model.state_dict()
    models_to_merge = []
    for k in ['b', 'c', 'd', 'e', 'f']:
        model = models[k]#.cuda()
        models_to_merge.append(model)
    
    exclude_param_names_regex = [
        'visual..*',
        '.*embed_tokens.*',
        '.*lm_head.*',
        '.*norm.*',
        '.*bias.*'
    ]
    
    if merge_method == "task_arithmetic":
        print("Running task_arithmetic...")
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "ties":
        print("Running ties_merging...")
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ta":
        print("Running Dare task_arithmetic...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ties":
        print("Running Dare ties_merging...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=new_models_to_merge, 
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "svd":
        print("Running SVD_merging...")
        merged_params = svd_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "iso":
        print("Running iso_merging...")
        merged_params = iso_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi":
        print("Running wudi_merging...")
        merged_params = wudi_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi2":
        print("Running wudi v2...")
        merged_params = wudi_merging2(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash":
        print("Running wudi_nash_merging...")
        merged_params = wudi_nash_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_shapley_value":
        print("Running wudi_shapley_value merging (Ablation: No Nash Optimization)...")
        merged_params = wudi_shapley_value(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_only_nash":
        print("Running wudi_only_nash merging (Ablation: No Shapley Weights)...")
        merged_params = wudi_only_nash(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "shapfed_nash":
        print("Running shapfed_nash_merging...")
        merged_params = shapfed_nash_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_core_nash":
        print("Running wudi_core_nash_merging...")
        merged_params = wudi_core_nash_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
            lora_rank=16 # 建议为16，和文献对齐，你可以随意修改
        )
    elif merge_method == "shapfed_core_nash":
        print("Running shapfed_core_nash_merging...")
        merged_params = shapfed_core_nash_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
            lora_rank=16
        )
    elif merge_method == "wudi_only_core":
        print("Running wudi_only_core_merging (Ablation: No Nash, No Shapley, Severe Rank Bottleneck)...")
        merged_params = wudi_only_core_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
            lora_rank=16
        )
    elif merge_method == "wudi_nash_lora_merging":
        print("Running wudi_merge_lora_merging...")
        merged_params = wudi_nash_lora_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
            lora_rank=16
        )
    elif merge_method == "wudi_nash_optmerging":
        print("Running wudi_nash_optmerging...")
        merged_params = wudi_nash_optmerging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")

    for key in merged_params:
        if key in base_state_dict:
            base_state_dict[key] = merged_params[key]
    base_model.load_state_dict(base_state_dict)
    base_model = base_model.cuda()

    if output_path is None:
        output_path = f'merged_model_{merge_method}' 
    print(f"Saving model to {output_path}")
    base_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()
    return base_model

#####################################################################
if __name__ == "__main__":
    path_a = 'Qwen/Qwen2-VL-7B'
    path_b = 'yongxianwei/Qwen2-VL-7B-OCR'
    path_c = 'yongxianwei/Qwen2-VL-7B-VQA'
    path_d = 'yongxianwei/Qwen2-VL-7B-Geometry'
    path_e = 'yongxianwei/Qwen2-VL-7B-Chart'
    path_f = 'yongxianwei/Qwen2-VL-7B-Grounding'

    processor = AutoProcessor.from_pretrained(path_a, trust_remote_code=True)
    models = {
        'a': Qwen2VLForConditionalGeneration.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'b': Qwen2VLForConditionalGeneration.from_pretrained(path_b, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'c': Qwen2VLForConditionalGeneration.from_pretrained(path_c, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'd': Qwen2VLForConditionalGeneration.from_pretrained(path_d, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'e': Qwen2VLForConditionalGeneration.from_pretrained(path_e, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'f': Qwen2VLForConditionalGeneration.from_pretrained(path_f, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    }
    
    original_base_state_dict = copy.deepcopy(models['a'].state_dict())
    # scaling_coefficients = [0.1,0.3, 0.5, 0.7, 0.9, 1.5]
    scaling_coefficients = [1.1,1.2,1.3,1.4]

    merge_method = "wudi_core_nash"
    for coeff in scaling_coefficients:
        print(f"\n=======================================================")
        print(f"Running experiment with scaling_coefficient={coeff}")
        print(f"=======================================================\n")
        
        # Reset base model to original state
        models['a'].load_state_dict(original_base_state_dict)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, f'merged_model_{merge_method}_coeff_{coeff}')
        
        # Call merging
        model = merge_models(merge_method=merge_method, scaling_coefficient=coeff, output_path=output_dir)

        #####################################################################
        # Set example image paths
        # image_path1 = 'examples/image1.jpg'
        # image_path2 = 'examples/image2.jpg'
        # import os
        # abs_image_path1 = os.path.abspath(image_path1)
        # abs_image_path2 = os.path.abspath(image_path2)

        # print("\n=== Single Image Example ===")
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": f"file://{abs_image_path1}"},
        #             {"type": "text", "text": "Please describe the image shortly."},
        #         ],
        #     }
        # ]

        # text = processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        # generation_config = dict(max_new_tokens=2048, do_sample=True)
        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(f"Model answer: {output_text[0]}")

        # print("\n=== Multi-turn Dialogue Example ===")
        # messages1 = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": f"file://{abs_image_path1}"},
        #             {"type": "text", "text": "Please describe the image in detail."},
        #         ],
        #     }
        # ]

        # text = processor.apply_chat_template(
        #     messages1, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages1)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # first_response = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        # print(f"Model answer {first_response}")

        # messages2 = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": f"file://{abs_image_path1}"},
        #             {"type": "text", "text": "Please describe the image in detail."},
        #         ],
        #     },
        #     {
        #         "role": "assistant",
        #         "content": first_response,
        #     },
        #     {
        #         "role": "user",
        #         "content": "Please write a poem according to the image.",
        #     }
        # ]

        # text = processor.apply_chat_template(
        #     messages2, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages2)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # poem_response = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        # print(f"Model answer (poem): {poem_response}")

        # print("\n=== Multiple Images Example ===")
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": f"file://{abs_image_path1}"},
        #             {"type": "image", "image": f"file://{abs_image_path2}"},
        #             {"type": "text", "text": "What are the similarities and differences between these two images."},
        #         ],
        #     }
        # ]

        # text = processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(f"Model answer: {output_text[0]}")

        # print("\n=== Pure Text Dialogue Example ===")
        # messages = [
        #     {
        #         "role": "user",
        #         "content": "Hello, who are you? Can you tell me a story?",
        #     }
        # ]

        # text = processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # inputs = processor(
        #     text=[text],
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(f"Model answer: {output_text[0]}")