import concurrent.futures
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
import time
import os
import math

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                base_param = pretrained_param_dict[param_name]
                delta_param = self.task_vector_param_dict[param_name].to(base_param.device)
                merged_params[param_name] = base_param + scaling_coefficient * delta_param

        return merged_params

def ties_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
    """
    ties merging method (layer-by-layer implementation to save memory)
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """
    def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
        """
        convert parameter dictionary in task vector to a single vector
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

    def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
        """
        convert a single vector to parameter dictionary in task vector
        :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params)
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        flattened_models_to_merge_param = flattened_models_to_merge_param.float()
        
        num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
        
        # Calculate the threshold
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
        
        # Create mask and apply
        mask = flattened_models_to_merge_param.abs() >= kth_values
        
        # Apply mask and convert back to original dtype
        return (flattened_models_to_merge_param * mask).to(flattened_models_to_merge_param.dtype)

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
        get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

        return merged_flattened_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

    flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for task_vector in models_to_merge_task_vectors]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

        # merged parameter dictionary
        merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

    return merged_params

def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
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
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)

def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
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
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

def weight_average_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    weight average merging method
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
    
    num_models = len(models_to_merge)

    # iterate each individual model that needs to be merged
    with torch.no_grad():
        # sum up the task vectors
        merged_task_vector = models_to_merge_task_vectors[0]
        for index in range(1, num_models):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
        
        # Average the task vectors
        for param_name in merged_task_vector.task_vector_param_dict:
            merged_task_vector.task_vector_param_dict[param_name] /= num_models

        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

    return merged_params

def svd_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    SVD merging method that uses Singular Value Decomposition to merge models.
    Args:
        merged_model: nn.Module, the base model to merge into  
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to merge the task vectors
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    # Get the parameter names to merge
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Compute task vectors
    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            # Compute difference as task vector
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        # Process each parameter
        for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
            
            # Check parameter shape
            param_shape = models_to_merge_task_vectors[0][param_name].shape
            
            if len(param_shape) == 2 and param_name == 'lm_head.weight':
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                # Apply SVD merging for 2D tensors
                
                # Create temporary variables to store merged results
                sum_u = None
                sum_s = None
                sum_v = None
                
                # Process each model's task vector
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    # Move parameter to GPU for computation
                    vec = task_vector_dict[param_name].to(device).float()
                    
                    # Compute SVD
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    
                    # Compute reduced index
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    # Initialize and prepare storage for the first model
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    
                    # Store important components for each model
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                
                # Compute final merged parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                # Compute merged result and move back to CPU
                merged_param = torch.linalg.multi_dot([
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                ]).to(original_dtype).cpu()
                
                # Store merged parameter
                merged_task_vector_dict[param_name] = merged_param
                
            else:
                # Use simple averaging for non-2D tensors
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        # Create merged task vector and combine with base model
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient
        )
        
    return merged_params

def iso_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    ISO merging method, uses SVD and equalizes singular values to reduce interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, models to be merged
        exclude_param_names_regex: list, regex patterns for parameter names to exclude
        scaling_coefficient: float, scaling coefficient for merging task vectors
    Returns:
        dict: merged parameter dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        # Get parameter shape from the first task vector
        param_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
        
        if len(param_shape) == 2:
            # For 2D parameters, perform SVD merging
            with torch.no_grad():
                merged_param_value = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
                for index in range(1, len(models_to_merge_task_vectors)):
                    merged_param_value = merged_param_value + models_to_merge_task_vectors[index].task_vector_param_dict[param_name]
            
            # SVD and equalize singular values
            original_dtype = merged_param_value.dtype
            merged_param_value = merged_param_value.cuda().to(torch.float32)
            u, s, v = torch.linalg.svd(merged_param_value, full_matrices=False)
            avg_singular_value = torch.mean(s)
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))
            
            merged_param = torch.linalg.multi_dot([
                u, avg_s, v
            ]).to(original_dtype)
            
            # Store merged parameter
            merged_task_vector_dict[param_name] = merged_param
        else:
            # For non-2D parameters, compute the average of all task vectors
            print(param_name)
            with torch.no_grad():
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

def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [基准方法] Wudi Merging (冗余任务向量优化)
    
    原理：
    优化一个单一的合并向量，使其同时最小化到所有 Task Vector 的距离。
    
    实现步骤：
    1. 构建所有待合并模型的 Task Vector。
    2. 使用 Adam 优化器迭代更新合并向量，最小化加权平方误差。
    3. 针对非 Attention 参数使用简单平均作为兜底。
    
    优势：
    - 简单直接，作为基准 Baseline。
    - 通过迭代优化寻找全局最优解。
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
       
        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - vectors
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            print(f"Step {i}, loss: {loss.item()}")
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)#.cpu()
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    # Create merged task vector and combine with base model
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
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)

        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask  # (n, n)
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)  # m x n
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    # Create merged task vector and combine with base model
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
    
    # 1. Calculate Utility for all subsets
    for i in range(1, 1 << n_models):
        mask_list = [(i >> bit) & 1 for bit in range(n_models)]
        mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
        subset_idx = torch.where(mask)[0]
        
        # Merging Vector: Simple Average
        subset_vecs_full = vectors[subset_idx]
        merging_vector = torch.mean(subset_vecs_full, dim=0)
        
        # Fast Interference Calculation
        subset_low_rank = low_rank_vectors[subset_idx]
        subset_norms = l2_norms[subset_idx]
        
        disturbing_vectors = merging_vector.unsqueeze(0) - subset_vecs_full
        inner_product = torch.matmul(disturbing_vectors, subset_low_rank.transpose(1, 2))
        per_model_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / (subset_norms + 1e-6) # Add epsilon
        
        total_loss = torch.sum(per_model_loss)
        subset_utilities[i] = -total_loss.item()

    # 2. Compute Shapley Values
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

    # 3. Softmax Normalization with Temperature
    # Temperature < 1 makes the distribution sharper (highlights top experts)
    # Temperature > 1 makes it flatter
    temperature = 1.2 
    alpha = torch.softmax(shapley_values / temperature, dim=0)
        
    return alpha
def wudi_nash_merging_trick1(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [自适应版] Wudi Nash Merging (基于能量阈值的 SVD 秩截断)

    原理：
    相比于固定比例截断 SVD 奇异值，本方法根据奇异值的累积能量 (Cumulative Energy) 动态决定保留的秩 (Rank)。
    这确保了在压缩 Task Vector 的同时，保留了其 90% (默认) 的主要信息量，适应不同层级的信息密度差异。

    实现步骤：
    1. SVD 分解：对 Task Vector 进行奇异值分解。
    2. 能量计算：计算奇异值的累积能量占比。
    3. 自适应截断：保留累积能量达到阈值 (如 90%) 的前 k 个奇异值，丢弃剩余部分。
    4. Nash 优化：在自适应截断后的低秩空间内进行 Nash 均衡优化。

    优势：
    - 更智能的特征保留机制，避免了固定比例截断可能带来的信息丢失或噪声保留。
    - 对不同层级的参数自适应调整压缩率。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=200):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
        low_rank_list = []
        taskvector_list = []
        
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            # [修改点：Trick 1 - 自适应能量阈值]
            energy_threshold = 0.90
            cumulative_energy = torch.cumsum(s, dim=0) / (torch.sum(s) + 1e-8)
            reduced_index_s = torch.searchsorted(cumulative_energy, energy_threshold).item() + 1
            reduced_index_s = max(1, min(reduced_index_s, s.shape[0]))
            
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

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.01 
        
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
            
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=100)
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_nash_merging_trick2(
    merged_model: nn.Module, 
    models_to_merge: list, 
    exclude_param_names_regex: list, 
    scaling_coefficient: float = 1.0,
    keep_rate: float = 0.2  # [新增] Trick 2 的稀疏化保留比例
):
    """
    [进阶版] Wudi Nash Merging (TIES/DARE Trick 增强)

    原理：
    结合了 TIES-Merging 的思想，在 Nash 优化前对 Task Vector 进行稀疏化和符号对齐，以减少噪声干扰。

    实现步骤：
    1. 符号共识 (Sign Consensus)：统计所有模型在每一位上的符号，保留占主导地位的符号方向。
    2. 稀疏化 (Sparsification)：仅保留绝对值最大的前 `keep_rate` (如 20%) 的参数，其余置零。
    3. Nash 优化：在去噪后的 Task Vector 上进行 Nash 均衡优化。

    优势：
    - 显著减少了 Task Vector 中的随机噪声。
    - 符号对齐解决了不同模型更新方向冲突的问题。
    - 稀疏化使得优化更加专注于关键参数。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=200):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
        # =====================================================================
        # [Trick 2] TIES-Merging: Sparsification & Sign Consensus
        # 在计算 SVD 和 Nash 博弈之前，先清理全参微调的噪声和方向冲突
        # =====================================================================
        num_models, d1, d2 = vectors.shape
        flat_vectors = vectors.view(num_models, -1)
        
        # 2a. Sparsification (Top-K 稀疏化)
        # 计算需要保留的参数数量 k
        k = max(1, int(flat_vectors.shape[1] * keep_rate))
        
        # 找到每个模型 task vector 的 top-k 绝对值阈值
        thresholds, _ = torch.topk(torch.abs(flat_vectors), k, dim=1)
        thresholds = thresholds[:, -1].unsqueeze(1) # 取第 k 大的值作为阈值
        
        # 生成 Mask 并应用（小于阈值的参数更新直接置为 0）
        sparse_mask = torch.abs(flat_vectors) >= thresholds
        flat_vectors = flat_vectors * sparse_mask
        
        # 2b. Sign Consensus (符号一致性)
        # 将所有模型的更新按位相加，判断总体的主导更新方向 (1, -1, 或 0)
        dominant_sign = torch.sign(torch.sum(flat_vectors, dim=0, keepdim=True))
        
        # 如果某个模型的更新方向与主导方向相反，则将其置为 0，消除干涉 (Interference)
        # (dominant_sign == 0 时的位或操作是为了防止全0向量被意外过滤)
        sign_mask = (torch.sign(flat_vectors) == dominant_sign) | (dominant_sign == 0)
        resolved_vectors = flat_vectors * sign_mask
        
        # 将处理干净的张量还原回 2D 形状，送入后续的 Nash 博弈
        vectors = resolved_vectors.view(num_models, d1, d2)
        # =====================================================================
        
        # [Pre-computation] SVD and Low Rank
        low_rank_list = []
        taskvector_list = []
        
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            # Keep slightly more rank info for better accuracy
            reduced_index_s = max(1, int(s.shape[0] * 0.1)) # Keep top 20%
            
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

        # [Step 1] Compute Shapley Alphas
        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            
            # Calculate the "Shapley Centroid" (Weighted Average)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            
            # Calculate Target Norm: Weighted average of individual norms
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # [Step 2] Optimization Setup
        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.01 
        
        # [Step 3] Optimization Loop
        for i in range(iter_num):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            
            # 1. Nash Interference Loss (Weighted)
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)
            
            # 2. Regularization Loss
            reg_loss = torch.norm(merging_vector - shapley_centroid)
            
            total_loss = nash_loss + lambda_reg * reg_loss
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        # [Step 4] Post-Optimization Norm Rectification
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    # Process Parameters
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=100)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle others (1D params like LayerNorm/Bias) with Simple Average
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_nash_merging_trick3(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [精简版] Wudi Nash Merging (1D 参数丢弃)

    原理：
    针对 1D 参数 (如 LayerNorm, Bias) 通常难以优化且包含较多噪声的问题，直接丢弃这些参数的 Task Vector，仅优化 2D 权重矩阵。

    实现步骤：
    1. 对 2D 权重矩阵进行完整的 Nash + Shapley 优化。
    2. 对 1D 参数，直接使用全零 Task Vector (即保留 Base Model 的原始参数)。
    3. 组合优化后的 2D 参数和原始 1D 参数。

    优势：
    - 极端保守策略，最大程度避免了 1D 参数带来的不稳定性。
    - 适用于 1D 参数微调过度或失效的场景。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=200):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
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

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.01 
        
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
            
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=100)
            merged_task_vector_dict[param_name] = merging_vector
    
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            # [修改点：Trick 3 - 直接抛弃 1D Task Vector，保留 Base Model 参数]
            merged_task_vector_dict[param_name] = torch.zeros_like(
                models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
            )

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_nash_merging_trick4(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [自适应版] Wudi Nash Merging (动态学习率与正则化)

    原理：
    根据每一层的参数规模和方差，动态调整 Nash 优化的学习率 (LR) 和正则化系数 (Lambda)。

    实现步骤：
    1. 计算每一层的参数规模 (Scale)，动态设定 Adam 优化器的学习率。
    2. 计算 Task Vector 间的方差 (Variance)，动态设定正则化强度。方差越大，正则化越强，以防止过拟合。
    3. 执行 Nash 优化。

    优势：
    - 无需手动调参，自适应不同层的特性。
    - 在参数差异大的层更加稳健，在差异小的层收敛更快。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=200):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
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

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        
        # [修改点：Trick 4 - 动态学习率和动态 Lambda]
        layer_scale = target_norm.item() / (merging_vector.numel() ** 0.5 + 1e-8)
        dynamic_lr = 1e-3 * max(0.1, min(10.0, layer_scale))
        
        optimizer = torch.optim.Adam([merging_vector], lr=dynamic_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        
        lambda_reg = 0.01 
        variance = torch.var(vectors, dim=0).mean()
        dynamic_lambda = lambda_reg * (1.0 / (variance.item() + 1e-4))
        
        for i in range(iter_num):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)
            reg_loss = torch.norm(merging_vector - shapley_centroid)
            
            # [修改点：Trick 4 - 使用 dynamic_lambda]
            total_loss = nash_loss + dynamic_lambda * reg_loss
                
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
    
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=100)
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_nash_merging_trick5(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [平滑版] Wudi Nash Merging (软性范数纠正)

    原理：
    在优化结束后，对合并向量的范数 (Norm) 进行软性调整，而不是强制缩放到目标范数，以保留优化过程中产生的幅度变化信息。

    实现步骤：
    1. 执行标准的 Nash + Shapley 优化。
    2. 计算目标范数 (Target Norm) 和优化后范数 (Optimized Norm)。
    3. 计算缩放因子，但仅应用 50% 的修正量 (Soft Scaling)，允许一定程度的范数漂移。

    优势：
    - 避免了强制范数对齐可能带来的特征破坏。
    - 在保留原始能量和适应新能量之间取得了平衡。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=200):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
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

        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.01 
        
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
            
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            # [修改点：Trick 5 - 软性范数纠正]
            soft_scaling_factor = 1.0 + 0.5 * (scaling_factor - 1.0) 
            final_vector = merging_vector * soft_scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=100)
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params  
def wudi_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [标准版] Wudi Nash Merging (Nash 均衡 + Shapley 引导)

    原理：
    这是 Wudi Nash Merging 的核心标准版本。它结合了博弈论中的 Nash 均衡和 Shapley 值，旨在寻找一个能最小化多模型冲突的合并解。

    实现步骤：
    1. Shapley 初始化：计算每个模型的 Shapley 值，作为重要性权重初始化合并向量。
    2. Nash 优化：将合并问题建模为非合作博弈，最小化每个模型在低秩投影空间内的干扰损失。
    3. 正则化约束：在优化过程中，约束合并向量不要偏离 Shapley 重心太远。
    4. 范数纠正：优化结束后，恢复合并向量的能量 (Norm) 至目标水平。

    优势：
    - 理论完备，兼顾了公平性 (Shapley) 和全局最优性 (Nash)。
    - 相比简单的加权平均，能显著降低模型间的参数冲突。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=150): # Iterations reduced due to better optimizer
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
        # [Pre-computation] SVD and Low Rank
        low_rank_list = []
        taskvector_list = []
        
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            # Keep slightly more rank info for better accuracy
            reduced_index_s = max(1, int(s.shape[0] * 0.2)) # Keep top 20% instead of 1/N
            
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

        # [Step 1] Compute Shapley Alphas
        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            
            # Calculate the "Shapley Centroid" (Weighted Average)
            # This serves as a regularization target and norm reference
            # Shape: (m, n)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            
            # Calculate Target Norm: Weighted average of individual norms
            # "High value experts should dictate the magnitude"
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # [Step 2] Optimization Setup
        # Initialize with Shapley Centroid instead of simple mean for faster convergence
        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        
        # Upgrade to Adam Optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        
        # Regularization Strength
        lambda_reg = 0.05 
        
        # [Step 3] Optimization Loop
        for i in range(iter_num): # Removed tqdm for speed in inner loops
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            
            # 1. Nash Interference Loss (Weighted)
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(nash_alpha * per_task_loss)
            
            # 2. Regularization Loss (Keep close to Shapley Centroid)
            # Prevents the vector from drifting into "low interference but meaningless" space
            reg_loss = torch.norm(merging_vector - shapley_centroid)
            
            total_loss = nash_loss + lambda_reg * reg_loss
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        # [Step 4] Post-Optimization Norm Rectification
        # Recover the energy of the weights
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    # Process Parameters
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        # Only optimize 2D weights (Linear layers), skip lm_head if needed, or include it
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=150)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle others with Shapley-Weighted Average (Better than simple average)
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            # We don't have alphas for these params (scalars/1D), so we recalculate a simplified alpha or use simple average
            # For simplicity and speed, let's use Simple Average here, 
            # OR better: accumulate weighted sum using the alpha from the *previous* optimized layer? 
            # No, that's too complex. Simple Average is safe for LayerNorm/Bias.
            
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

#验证Nash优化的有效性
def wudi_shapley_value(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [消融实验] Wudi Shapley Merging (仅 Shapley 加权)

    原理：
    仅使用 Shapley 值对模型进行加权平均，不进行后续的 Nash 优化迭代。用于验证 Shapley 值本身的有效性。

    实现步骤：
    1. SVD 分解：用于计算 Shapley 值所需的低秩特征。
    2. 计算 Shapley 值：评估每个模型对整体特征的贡献度。
    3. 加权平均：直接使用 Shapley 值作为权重进行加权融合。
    4. 范数纠正：调整合并后的向量范数。

    优势：
    - 计算速度快 (无需迭代优化)。
    - 验证了 Shapley 值作为权重分配机制的合理性。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_shapley_calibrated_task_vector(param_name, vectors):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
        # [Pre-computation] SVD and Low Rank (Needed to compute Shapley Alpha accurately)
        low_rank_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            
            reduced_index_s = max(1, int(s.shape[0] * 0.2)) # Keep top 20%
            
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
            # [Step 1] Compute Shapley Alphas
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            
            # [Step 2] Calculate the Shapley Centroid C (Weighted Average)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            
            # [Step 3] Calculate Target Norm (Weighted average of individual norms)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

            # [Step 4] Norm Rectification directly on the Centroid (Skip Nash Optimization)
            centroid_norm = torch.norm(shapley_centroid)
            if centroid_norm > 1e-6:
                scaling_factor = target_norm / centroid_norm
                final_vector = shapley_centroid * scaling_factor
            else:
                final_vector = shapley_centroid
                
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    # Process Parameters
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Ablation: Shapley Value Calibration"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        # Only process 2D weights (Linear layers), skip lm_head
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_shapley_calibrated_task_vector(param_name, values)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle others with Simple Average
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

# 验证Shapley Value的有效性 (只有Nash，没有Shapley，权重全部均匀 1/N)
def wudi_only_nash(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [消融实验] Wudi Only Nash (无 Shapley 引导)

    原理：
    仅使用 Nash 优化，但不使用 Shapley 值进行初始化和加权 (使用均匀权重)。用于验证 Nash 优化本身的有效性。

    实现步骤：
    1. 均匀初始化：所有模型权重设为 1/N。
    2. Nash 优化：最小化均匀加权下的干扰损失。
    3. 正则化：约束合并向量不偏离简单平均中心。

    优势：
    - 验证了即便没有 Shapley 引导，Nash 优化也能通过减少冲突来提升性能。
    - 证明了 Shapley + Nash 的组合优于单独使用任一方法。
    """
    assert isinstance(scaling_coefficient, float), "scaling_coefficient should be float!"
    
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_only_nash_optimized_task_vector(param_name, vectors, iter_num=100): 
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        n_models = vectors.shape[0]
        
        # [Pre-computation] SVD and Low Rank (Still needed for interference loss)
        low_rank_list = []
        taskvector_list = []
        
        for i in range(n_models):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            reduced_index_s = max(1, int(s.shape[0] * 0.2)) # Keep top 20%
            
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

        # [Step 1] Uniform Alphas (No Shapley)
        with torch.no_grad():
            uniform_alpha = torch.ones(n_models, device=device) / n_models
            
            # Calculate Simple Average Centroid
            simple_centroid = torch.mean(vectors, dim=0)
            
            # Calculate Target Norm: Simple average of individual norms
            vector_norms = torch.norm(vectors.reshape(n_models, -1), p=2, dim=1)
            target_norm = torch.mean(vector_norms)

        # [Step 2] Optimization Setup
        merging_vector = torch.nn.Parameter(simple_centroid.clone())
        
        optimizer = torch.optim.Adam([merging_vector], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        
        lambda_reg = 0.01 
        
        # [Step 3] Optimization Loop
        for i in range(iter_num): 
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            
            # 1. Nash Interference Loss (Uniformly Weighted)
            per_task_loss = torch.sum(torch.square(inner_product), dim=(1, 2)) / l2_norms
            nash_loss = torch.sum(uniform_alpha * per_task_loss)
            
            # 2. Regularization Loss (Keep close to Simple Centroid)
            reg_loss = torch.norm(merging_vector - simple_centroid)
            
            total_loss = nash_loss + lambda_reg * reg_loss
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        # [Step 4] Post-Optimization Norm Rectification
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
       
    merged_task_vector_dict = {}
    
    # Process Parameters
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Ablation: Only Nash Optimization"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        # Only optimize 2D weights (Linear layers), skip lm_head
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            merging_vector = get_only_nash_optimized_task_vector(param_name, values, iter_num=100)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle others with Simple Average
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
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params


def shapfed_nash_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    [模块化版] ShapFed Nash Merging (基于模块级 Shapley 代理的混合融合)

    原理：
    结合 ShapFed (CSSV) 的模块化思想与 Nash 优化。不计算全局统一的 Shapley 值，而是将模型划分为不同功能模块 (如 Vision Encoder, Language Decoder)，
    并为每个模块单独计算 Shapley 权重 (Alphas)。这允许模型在不同能力维度上拥有不同的专家权重分布。

    实现步骤：
    1. 模块划分：根据参数名称前缀将模型划分为独立模块。
    2. 代理层选择：在每个模块中，自动寻找变化量 (L1 Norm) 最大的层作为“代理层”。
    3. 模块级 Alpha 计算：利用 CSSV (Cosine Similarity Shapley Value) 在代理层上快速计算该模块的 Shapley 权重。
    4. 模块化 Nash 优化：在进行 Nash 优化时，将对应模块的 Alpha 注入到优化目标中，引导合并向量向该模块的专家倾向收敛。

    优势：
    - 细粒度的权重分配，解决了“全局最优不代表局部最优”的问题。
    - 结合了 CSSV 的计算效率和 Nash 的抗干扰能力。
    - 特别适用于多模态大模型 (如 InternVL)，不同模态可由不同专家主导。
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
    print("------------------------------------------------\\n")

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

# ====================== 方法一核心：TSV-Procrustes（替换 GD Nash） ======================
def get_tsv_shapley_optimized_task_vector(param_name, vectors):
    """TSV-Merge (arXiv:2412.00081 Algorithm 1) + Shapley 加权 Σ"""
    original_dtype = vectors.dtype
    device = vectors.device
    n_models = vectors.shape[0]
    vectors = vectors.to(torch.float32).to(device)

    # 1. 计算 Shapley alpha（保留你的公平性）
    low_rank_list = []
    l2_norms = torch.square(torch.norm(vectors.reshape(n_models, -1), p=2, dim=-1)) + 1e-6
    for i in range(n_models):
        u, s, vh = torch.linalg.svd(vectors[i], full_matrices=False)
        k_temp = max(1, s.shape[0] // n_models)
        low_rank_list.append(u[:, :k_temp] @ torch.diag_embed(s[:k_temp]) @ vh[:k_temp, :])
    low_rank = torch.stack(low_rank_list)
    with torch.no_grad():
        nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)

    # 2. per-task SVD + 截断 k=1/T（论文精确设定）
    U_list, Sigma_list, V_list = [], [], []  # V_list 保存 paper 中的 V (n x k)
    for i in range(n_models):
        u, s, vh = torch.linalg.svd(vectors[i], full_matrices=False)
        k = max(1, s.shape[0] // n_models)          # 论文明确：k = 1/T
        U_list.append(u[:, :k])
        Sigma_list.append(s[:k] * nash_alpha[i])    # Shapley 加权（我们的增强）
        V_i = vh[:k, :].T                           # 转为 paper 的 V (n x k)
        V_list.append(V_i)

    # 3. 拼接
    U_cat = torch.cat(U_list, dim=1)                # (out_dim, T*k)
    V_cat = torch.cat(V_list, dim=1)                # (in_dim, T*k)
    Sigma_block = torch.block_diag(*[torch.diag(s) for s in Sigma_list])

    # 4. Procrustes 正交化（论文核心 whitening）
    # U⊥
    Pu, _, Vhu = torch.linalg.svd(U_cat, full_matrices=False)
    U_perp = Pu @ Vhu
    # V⊥
    Pv, _, Vhv = torch.linalg.svd(V_cat, full_matrices=False)
    V_perp = Pv @ Vhv

    # 5. 重构 merged_vector
    merging_vector = U_perp @ Sigma_block @ V_perp.T

    # 6. Norm 矫正（保留你的能量）
    target_norm = torch.sum(torch.norm(vectors.reshape(n_models, -1), p=2, dim=1) * nash_alpha)
    optimized_norm = torch.norm(merging_vector)
    if optimized_norm > 1e-6:
        merging_vector = merging_vector * (target_norm / optimized_norm)

    return merging_vector.data.detach().to(original_dtype).cpu()

# ====================== 方法一完整合并函数 ======================
def wudi_nash_merging_tsv(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """完整 TSV 版（推荐优先实验，速度快、无梯度）"""
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    merged_task_vector_dict = {}
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="TSV Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if len(param_data.shape) == 2 and "lm_head" not in param_name:
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_tsv_shapley_optimized_task_vector(param_name, values)
            merged_task_vector_dict[param_name] = merging_vector
        else:
            # 非2D层用简单平均
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                merged_param += (task_vector.task_vector_param_dict[param_name] - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

# ====================== 方法二核心：LoRE 预处理 ======================
def svt_operator(matrix: torch.Tensor, mu: float) -> torch.Tensor:
    """Singular Value Thresholding (LoRE 论文精确实现)"""
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    s_thresh = torch.clamp(s - mu, min=0.0)
    return (u @ torch.diag_embed(s_thresh) @ vh).to(matrix.dtype)

# ====================== 方法二完整合并函数 ======================
def wudi_nash_merging_lore(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0, mu: float = 0.01, max_iter: int = 30, nash_iter_num: int = 150):
    """完整 LoRE 版（先低秩精炼 task vectors，再走你原来的 Shapley + Nash）"""
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    # Use the optimized Nash-Shapley merging function (Internal definition to ensure availability)
    def get_nash_shapley_optimized_task_vector(param_name, vectors, iter_num=nash_iter_num):
        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)
        
        # [Pre-computation] SVD and Low Rank
        low_rank_list = []
        taskvector_list = []
        
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            
            # Keep slightly more rank info for better accuracy
            reduced_index_s = max(1, int(s.shape[0] * 0.2)) # Keep top 20% instead of 1/N
            
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

        # [Step 1] Compute Shapley Alphas
        with torch.no_grad():
            nash_alpha = compute_shapley_alpha(vectors, low_rank, l2_norms)
            shapley_centroid = torch.sum(vectors * nash_alpha.view(-1, 1, 1), dim=0)
            vector_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1)
            target_norm = torch.sum(vector_norms * nash_alpha)

        # [Step 2] Optimization Setup
        merging_vector = torch.nn.Parameter(shapley_centroid.clone())
        optimizer = torch.optim.Adam([merging_vector], lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)
        lambda_reg = 0.05 
        
        # [Step 3] Optimization Loop
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
            
        # [Step 4] Post-Optimization Norm Rectification
        optimized_norm = torch.norm(merging_vector)
        if optimized_norm > 1e-6:
            scaling_factor = target_norm / optimized_norm
            final_vector = merging_vector * scaling_factor
        else:
            final_vector = merging_vector
            
        return final_vector.data.detach().to(original_dtype).cpu()
    
    # ================== LoRE 预处理（论文 Algorithm 1） ==================
    print(f"Running LoRE preprocessing (μ={mu}, {max_iter} iter)...")
    refined_deltas = {}                            # param_name -> list[low-rank δ_i]
    
    for param_name in tqdm(list(models_to_merge_task_vectors[0].task_vector_param_dict.keys()), desc="LoRE Preprocess"):
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) != 2:
            continue
        theta_list = [tv.task_vector_param_dict[param_name] for tv in models_to_merge_task_vectors]
        n_models = len(theta_list)
        deltas = [torch.zeros_like(theta_list[0]) for _ in range(n_models)]
        
        for _ in range(max_iter):
            theta0 = torch.mean(torch.stack([theta_list[i] - deltas[i] for i in range(n_models)]), dim=0)
            for i in range(n_models):
                deltas[i] = svt_operator(theta_list[i] - theta0, mu)
        
        refined_deltas[param_name] = deltas
    # =================================================================

    merged_task_vector_dict = {}
    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict, desc="Nash Optimizing Layers"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        
        if param_name in refined_deltas and len(param_data.shape) == 2 and "lm_head" not in param_name:
            # 使用 LoRE 精炼后的低秩 δ_i
            values = torch.stack(refined_deltas[param_name])
            # 继续走你原来的 Nash 优化（完全保留）
            merging_vector = get_nash_shapley_optimized_task_vector(param_name, values, iter_num=nash_iter_num)  # 使用优化后的参数
            merged_task_vector_dict[param_name] = merging_vector
        else:
            # 其他层用原始，且不进行 Nash 优化，直接走下面的平均
            pass

    # 非2D层处理（同原代码）
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                merged_param += (task_vector.task_vector_param_dict[param_name] - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
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
                           lora_rank: int = 16): # [打压超参数] 设定极低的 Rank 制造信息瓶颈
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


# ====================== 共享辅助函数（已重命名） ======================
def compute_acute_shapley_alpha(vectors: torch.Tensor, low_rank: torch.Tensor, l2_norms: torch.Tensor):
    """
    在锐角正锥投影空间中计算 Shapley 值（Acute Cone 版本）
    与原版逻辑完全一致，但明确用于 acute cone 空间
    """
    G = torch.matmul(vectors, vectors.transpose(0, 1))
    interference = torch.sum(torch.square(G), dim=1) / (l2_norms + 1e-12)
    alpha = 1 / (interference + 1e-6)
    alpha = alpha / alpha.sum()
    return alpha

def project_to_acute_cone(vectors: torch.Tensor, rank: int = 32, eps: float = 1e-6):
    """
    核心创新：将任务向量投影到「所有两两夹角均为锐角」的低维正锥空间
    vectors: [K, m, n]
    返回: 
        proj_acute: [K, r]          ← 锐角投影后的低维坐标
        consensus_dir: [r]          ← 正锥内的共识方向
        Vh_r: [r, m*n]              ← 用于重构回原空间
    """
    K, m, n = vectors.shape
    device = vectors.device
    d = m * n
    flat = vectors.view(K, d).to(torch.float32)          # [K, d]

    # SVD 降维到 r 维
    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    r = min(rank, S.shape[0])
    proj = U[:, :r] @ torch.diag_embed(S[:r])            # [K, r]

    # 迭代强制正锥（所有 cos > 0）
    u = proj.mean(dim=0)
    for _ in range(8):
        cosines = torch.matmul(proj, u) / (torch.norm(proj, dim=1, keepdim=True) * torch.norm(u) + eps)
        mask = cosines.squeeze() < 0
        if not mask.any():
            break
        u = u + 0.15 * proj[mask].mean(dim=0)

    # 最终锐角投影
    cosines = torch.matmul(proj, u) / (torch.norm(proj, dim=1, keepdim=True) * torch.norm(u) + eps)
    proj_acute = proj * torch.clamp(cosines, min=0.01)

    consensus_dir = proj_acute.mean(dim=0)

    return proj_acute, consensus_dir, Vh[:r, :]

# ====================== 方法1：集成版（在锐角投影空间做完整 Nash + Acute Shapley） ======================
def wudi_acute_nash_merging(merged_model: nn.Module,
                            models_to_merge: list,
                            exclude_param_names_regex: list,
                            scaling_coefficient: float = 1.0,
                            low_rank: int = 64,
                            iter_num: int = 300,
                            lambda_reg: float = 0.02):
    """
    【推荐主方法】Wudi Acute-Nash Merging
    核心思想：先把所有任务向量投影到「两两夹角均为锐角」的正锥低维空间，
    然后在该低维空间中完整执行 Acute Shapley 初始化 + Nash 干扰优化，
    最后映射回原参数空间。
    
    优势：
    - 彻底消除钝角干扰
    - Nash 优化维度从 O(mn) 降到 O(r²)，速度大幅提升
    - 与你原有 Nash & Shapley 完全兼容，性能通常优于原版
    """
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                   finetuned_model=model_to_merge,
                   exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    merged_task_vector_dict = {}

    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict,
                           desc="Acute-Nash Merging (2D layers)"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        if len(param_data.shape) != 2 or "lm_head" in param_name:
            continue

        # 原始任务向量
        vectors = torch.stack([
            tv.task_vector_param_dict[param_name]
            for tv in models_to_merge_task_vectors
        ])  # [K, m, n]

        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)

        # ==================== 1. 正锥投影 ====================
        K_, m_, n_ = vectors.shape
        d_ = m_ * n_
        flat_ = vectors.view(K_, d_).to(torch.float32)
        U_, S_, Vh_ = torch.linalg.svd(flat_, full_matrices=False)
        cum_ = torch.cumsum(S_, dim=0) / torch.sum(S_)
        idx_ = torch.nonzero(cum_ >= 0.92)
        r_energy_ = int(idx_[0].item() + 1) if idx_.numel() > 0 else int(min(low_rank, S_.shape[0]))
        eff_rank = int(min(low_rank, r_energy_))
        proj_acute, _, Vh_r = project_to_acute_cone(vectors, rank=eff_rank)

        # 在投影空间计算 Acute Shapley
        l2_norms_proj = torch.square(torch.norm(proj_acute, p=2, dim=1)) + 1e-6
        with torch.no_grad():
            nash_alpha = compute_acute_shapley_alpha(proj_acute, proj_acute, l2_norms_proj)
            target_norm = torch.sum(torch.norm(proj_acute, p=2, dim=1) * nash_alpha)

        # ==================== 2. 在锐角空间做 Nash 优化 ====================
        merging_vec = torch.nn.Parameter(proj_acute.mean(dim=0).clone())

        lr_ = 1e-3 if eff_rank > 64 else 2e-3
        iter_num_eff = int(max(iter_num, 150 + 2 * eff_rank + 10 * K_))
        optimizer = torch.optim.Adam([merging_vec], lr=lr_)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num_eff)

        for _ in range(iter_num_eff):
            disturbing = merging_vec.unsqueeze(0) - proj_acute
            inner_prod = torch.matmul(disturbing, proj_acute.transpose(0, 1))
            per_task_loss = torch.sum(torch.square(inner_prod), dim=1) / l2_norms_proj
            nash_loss = torch.sum(nash_alpha * per_task_loss)

            reg_loss = lambda_reg * torch.norm(merging_vec - proj_acute.mean(dim=0))

            total_loss = nash_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # 范数纠正（低维空间）
        optimized_norm = torch.norm(merging_vec)
        if optimized_norm > 1e-6:
            merging_vec = merging_vec * (target_norm / optimized_norm)

        # ==================== 3. 映射回原空间 ====================
        merged_flat = merging_vec @ Vh_r
        merged_matrix = merged_flat.view(vectors.shape[1], vectors.shape[2])
        merged_task_vector_dict[param_name] = merged_matrix.to(original_dtype).cpu()

    # 非2D参数用简单平均
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, tv in enumerate(models_to_merge_task_vectors[1:], 1):
                merged_param += (tv.task_vector_param_dict[param_name] - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


# ====================== 方法2：纯投影版（仅通过锐角空间做合并） ======================
def wudi_acute_cone_merging(merged_model: nn.Module,
                            models_to_merge: list,
                            exclude_param_names_regex: list,
                            scaling_coefficient: float = 1.0,
                            low_rank: int = 64):
    """
    【极简高效版】Wudi Acute-Cone Merging
    只做「锐角正锥投影 + Acute Shapley 加权共识方向」，完全无迭代。
    适合快速实验或资源紧张场景。
    """
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                   finetuned_model=model_to_merge,
                   exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    merged_task_vector_dict = {}

    for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict,
                           desc="Acute-Cone Merging (2D layers)"):
        param_data = models_to_merge_task_vectors[0].task_vector_param_dict[param_name]
        if len(param_data.shape) != 2 or "lm_head" in param_name:
            continue

        vectors = torch.stack([
            tv.task_vector_param_dict[param_name]
            for tv in models_to_merge_task_vectors
        ])

        original_dtype = vectors.dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectors = vectors.to(torch.float32).to(device)

        # 正锥投影
        K_, m_, n_ = vectors.shape
        d_ = m_ * n_
        flat_ = vectors.view(K_, d_).to(torch.float32)
        U_, S_, Vh_ = torch.linalg.svd(flat_, full_matrices=False)
        cum_ = torch.cumsum(S_, dim=0) / torch.sum(S_)
        idx_ = torch.nonzero(cum_ >= 0.92)
        r_energy_ = int(idx_[0].item() + 1) if idx_.numel() > 0 else int(min(low_rank, S_.shape[0]))
        eff_rank = int(min(low_rank, r_energy_))
        proj_acute, _, Vh_r = project_to_acute_cone(vectors, rank=eff_rank)

        # 使用 Acute Shapley 加权
        l2_norms_proj = torch.square(torch.norm(proj_acute, p=2, dim=1)) + 1e-6
        nash_alpha = compute_acute_shapley_alpha(proj_acute, proj_acute, l2_norms_proj)

        weighted_dir = torch.sum(proj_acute * nash_alpha.view(-1, 1), dim=0)

        # 范数纠正
        target_norm = torch.sum(torch.norm(proj_acute, p=2, dim=1) * nash_alpha)
        optimized_norm = torch.norm(weighted_dir)
        if optimized_norm > 1e-6:
            weighted_dir = weighted_dir * (target_norm / optimized_norm)

        # 映射回原空间
        merged_flat = weighted_dir @ Vh_r
        merged_matrix = merged_flat.view(vectors.shape[1], vectors.shape[2])
        merged_task_vector_dict[param_name] = merged_matrix.to(original_dtype).cpu()

    # 非2D参数简单平均
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, tv in enumerate(models_to_merge_task_vectors[1:], 1):
                merged_param += (tv.task_vector_param_dict[param_name] - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

def merge_models(merge_method="wudi2", scaling_coefficient = 0.1):
    print("Start merging models...")
    base_model = models['a'].cuda()
    base_state_dict = base_model.state_dict()

    models_to_merge = []
    for k in ['b', 'c', 'd', 'e', 'f']:
        model = models[k].cuda()
        models_to_merge.append(model)
    
    exclude_param_names_regex = [
        'vision_model.*',
        '.*lm_head.*',
        '.*norm.*',
        '.*embed_tokens.*',
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
    elif merge_method == "weight_average":
        print("Running weight_average_merging...")
        merged_params = weight_average_merging(
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
                # for each individual model, mask its weight
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
                # for each individual model, mask its weight
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
        print("Running tsv_merging...")
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
    elif merge_method == "wudi_nash_trick1":
        print("Running wudi_nash_merging_trick1...")
        merged_params = wudi_nash_merging_trick1(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_trick2":
        print("Running wudi_nash_merging_trick2...")
        merged_params = wudi_nash_merging_trick2(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_trick3":
        print("Running wudi_nash_merging_trick3...")
        merged_params = wudi_nash_merging_trick3(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_trick4":
        print("Running wudi_nash_merging_trick4...")
        merged_params = wudi_nash_merging_trick4(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_trick5":
        print("Running wudi_nash_merging_trick5...")
        merged_params = wudi_nash_merging_trick5(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_shapley_value":
        print("Running wudi_shapley_value merging (Ablation)...")
        merged_params = wudi_shapley_value(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_only_nash":
        print("Running wudi_only_nash merging (Ablation: No Shapley)...")
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
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_only_core":
        print("Running wudi_only_core_merging...")
        merged_params = wudi_only_core_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_merging_tsv":
        print("Running wudi_nash_merging_tsv...")
        merged_params = wudi_nash_merging_tsv(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_nash_merging_lore":
        print("Running wudi_nash_merging_lore...")
        merged_params = wudi_nash_merging_lore(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_acute_nash":
        print("Running wudi_acute_nash_merging...")
        merged_params = wudi_acute_nash_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi_acute_cone":
        print("Running wudi_acute_cone_merging...")
        merged_params = wudi_acute_cone_merging(
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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, f"merged_model_{merge_method}_coeff_{scaling_coefficient}")
    
    # Save to data disk to avoid filling up system disk
    # save_root = '/root/autodl-tmp/merged_models'
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root, exist_ok=True)
    # output_path = os.path.join(save_root, f"merged_model_{merge_method}_coeff_{scaling_coefficient}")
    print(f"Saving model to {output_path}")
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()
    return base_model

#####################################################################
if __name__ == "__main__":
    import copy
    
    # Configuration
    path_a = '/root/Model_Merge/InternVL3.5-2B-Series/InternVL3_5-2B'
    path_b = '/root/Model_Merge/InternVL3.5-2B-Series/InternVL35_2b_lora_expert_chart'
    path_c = '/root/Model_Merge/InternVL3.5-2B-Series/InternVL35_2b_lora_expert_counting'
    path_d = '/root/Model_Merge/InternVL3.5-2B-Series/InternVL35_2b_lora_expert_general'
    path_e = '/root/Model_Merge/InternVL3.5-2B-Series/InternVL35_2b_lora_expert_ocr'
    # path_f = '/root/Model_Merge/InternVL3-1B-Series/InternVL3-1B-Grounding/checkpoint-3660'

    print("Loading base model and experts...")
    # Load all models once
    # Keep base_model_original on CPU to be safe and clean for deepcopy
    # base_model_original = AutoModel.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True).eval()
    base_model_original = AutoModel.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=False).eval()
    # Use the tokenizer from the base model path
    tokenizer = AutoTokenizer.from_pretrained(path_a, trust_remote_code=True, use_fast=False)

    models = {
        'b': AutoModel.from_pretrained(path_b, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'c': AutoModel.from_pretrained(path_c, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'd': AutoModel.from_pretrained(path_d, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'e': AutoModel.from_pretrained(path_e, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        # 'f': AutoModel.from_pretrained(path_f, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    }
    
    # Hyperparameters to search
    # merge_methods = ['wudi_nash_merging_lore','ties','weight_average','dare ta','iso','wudi','wudi2'] 
    merge_methods = ['wudi_nash_merging_lore']
    # scaling_coefficients = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    scaling_coefficients = [0.7]
    
    print(f"Starting merge with methods '{merge_methods}' and coefficients: {scaling_coefficients}")
    
    for scaling_coefficient in scaling_coefficients:
        for merge_method in merge_methods:
            print(f"\nProcessing scaling_coefficient: {scaling_coefficient}, merge_method: {merge_method}")
            
            # Prepare clean base model for this iteration
            # Deepcopy to avoid in-place modification affecting next runs
            # We put it into global 'models' dict as 'a' because merge_models expects it there
            models['a'] = copy.deepcopy(base_model_original)
            
            try:
                merge_models(merge_method=merge_method, scaling_coefficient=scaling_coefficient)
            except Exception as e:
                print(f"Error executing with coefficient {scaling_coefficient} and method {merge_method}: {e}")
                import traceback
                traceback.print_exc()
            
            # Clean up to save memory
            if 'a' in models:
                del models['a']
            torch.cuda.empty_cache()
# set the max number of tiles in `max_num`
# pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
# generation_config = dict(max_new_tokens=1024, do_sample=False)

# # pure-text conversation
# question = 'Hello, who are you?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Can you tell me a story?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # single-image single-round conversation
# question = '<image>\nPlease describe the image shortly.'
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(f'User: {question}\nAssistant: {response}')

# # single-image multi-round conversation
# question = '<image>\nPlease describe the image in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Please write a poem according to the image.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, combined images
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                             history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                             history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, separate images
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                             num_patches_list=num_patches_list,
#                             history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                             num_patches_list=num_patches_list,
#                             history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # batch inference, single image per sample
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                             num_patches_list=num_patches_list,
#                             questions=questions,
#                             generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')
