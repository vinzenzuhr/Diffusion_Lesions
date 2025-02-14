"""
A large part of the code regarding the function "collate" originally comes from the official 
PyTorch implementation 'pytorch/pytorch/torch/utils/data/_utils/collate.py' received from Github
"""

import collections
import copy
from typing import Optional, Dict, Union, Type, Tuple, Callable
 
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data._utils.collate import default_collate_err_msg_format, default_collate_fn_map, default_collate 

from custom_modules import DatasetMRI

def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None, 
            multi_sample = False):
    r"""
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. 
    `default_collate_fn_map` provides default collate functions for tensors, numpy arrays, numbers 
    and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate 
            function. If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        >>> # Extend this function to handle batch of tensors
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem) 

    if collate_fn_map is not None:
        # if multi_sample then we have already a prepared batch 
        if multi_sample and elem_type == torch.Tensor:  
            assert len(batch)==1, "if multi_sample is active batch containing one multi_sample element is expected" 
            return batch[0]
        
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                clone.update({
                    key: collate(
                        [d[key] for d in batch], 
                        collate_fn_map=collate_fn_map, 
                        multi_sample=multi_sample
                    ) 
                    for key in elem
                })
                return clone
            else:
                return elem_type({
                    key: collate(
                        [d[key] for d in batch], 
                        collate_fn_map=collate_fn_map, 
                        multi_sample=multi_sample
                    ) 
                    for key in elem
                })
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate(
                    [d[key] for d in batch], 
                    collate_fn_map=collate_fn_map, 
                    multi_sample=multi_sample
                    ) 
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(
            collate(samples, collate_fn_map=collate_fn_map, multi_sample=multi_sample) 
            for samples in zip(*batch)
        ))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map, multi_sample=multi_sample) 
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(
                            samples, collate_fn_map=collate_fn_map, multi_sample=multi_sample)
                    return clone
                else:
                    return elem_type([
                        collate(samples, collate_fn_map=collate_fn_map, multi_sample=multi_sample) 
                        for samples in transposed
                    ])
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map, multi_sample=multi_sample) 
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type)) 

def collate_multi_sample(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map, multi_sample=True)

def get_dataloader(
        dataset: DatasetMRI, 
        batch_size: int, 
        num_workers: int = 8, 
        random_sampler: bool = False, 
        seed: int = 0, 
        multi_slice: bool = False) -> DataLoader:
    """
    Returns a PyTorch DataLoader object for the given dataset. 

    Args:
        dataset (DatasetMRI): The dataset to load.
        batch_size (int): The batch size for each iteration.
        num_workers (int, optional): The number of worker threads to use for data loading. 
            Defaults to 8.
        random_sampler (bool, optional): Whether to seed the random sampler for the dataset. 
            Defaults to False.
        seed (int, optional): The seed value for the random sampler. Defaults to 0.
        multi_slice (bool, optional): A boolean indicating whether the dataset contains
            more than one slice per sample. This would be the case for the pseudo3Dmodels, 
            where the model expects sorted slices within one sample. Defaults to False. 

    Returns:
        DataLoader: A PyTorch DataLoader object for the given dataset.
    """

    sampler = RandomSampler(
        dataset, generator=(None if random_sampler else torch.Generator().manual_seed(seed)))
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                    sampler=sampler, pin_memory=True, 
                    collate_fn=collate_multi_sample if multi_slice else default_collate)
    return dl
