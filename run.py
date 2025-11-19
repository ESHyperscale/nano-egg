import os

import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import operator

import tqdm

import tyro
from dataclasses import dataclass

from datasets import load_dataset

import numpy as np
import scipy
import time

import wandb
from pathlib import Path

from typing import Optional, NamedTuple

from functools import partial

import pickle


@dataclass
class Args:
    seed: int = 0

    dtype: str = 'int8'
    parallel_generations_per_gpu: int = 1024
    num_epochs: int = 1000

    alpha: float = 4.0
    sigma_shift: int = 4
    use_clt: bool = True
    fast_fitness: bool = True

    n_layer: int = 6
    n_embd: int = 256
    vocab_size: int = 256

    noise_reuse: int = 1
    tokens_per_update: int = 100

    num_perturbations: int = 0

    group_size: int = 2

    dir_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cached_files")
    train_output_path: str = "minipile_train.npy"
    valid_output_path: str = "minipile_valid.npy"
    test_output_path: str = "minipile_test.npy"
    
    regenerate_model: bool = False

    wandb_project: str = "HyperscalePretraining"
    tag: str = ""
    track: bool = False

    validate_every: int = 0
    validation_batch_size: int = 1024

args = tyro.cli(Args)


PARAM = 0
MM_PARAM = 1
EMB_PARAM = 2

FIXED_POINT = 4

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'silu': jax.nn.silu,
    'pqn': lambda x: jax.nn.relu(layer_norm(x))
}

DTYPE = jnp.dtype(args.dtype)
MAX = jnp.iinfo(DTYPE).max
LOGMAX = 7

suffix = ".model"

def save(model: any, path: str | Path, overwrite: bool = False):
    # See https://github.com/google/jax/issues/2116#issuecomment-580322624
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load(path: str | Path) -> any:
    # See https://github.com/google/jax/issues/2116#issuecomment-580322624
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data

### MODEL IMPLEMENTATION

class CommonInit(NamedTuple):
    frozen_params: any
    params: any
    scan_map: any
    es_map: any

class CommonParams(NamedTuple):
    noiser: any
    frozen_noiser_params: any
    noiser_params: any
    frozen_params: any
    params: any
    es_tree_key: any
    iterinfo: any

def recursive_scan_split(param, base_key, scan_tuple):
    # scan_tuple = tuple() implies no split
    if len(scan_tuple) == 0:
        return base_key
    # otherwise, it is (0, 1, ...)
    split_keys = jax.random.split(base_key, param.shape[scan_tuple[0]])
    return jax.vmap(recursive_scan_split, in_axes=(None, 0, None))(param, split_keys, scan_tuple[1:])

def simple_es_tree_key(params, base_key, scan_map):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(base_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    return jax.tree.map(recursive_scan_split, params, partial_key_tree, scan_map)

def merge_inits(**kwargs):
    params = {}
    frozen_params = {}
    scan_map = {}
    es_map = {}
    for k in kwargs:
        params[k] = kwargs[k].params
        scan_map[k] = kwargs[k].scan_map
        es_map[k] = kwargs[k].es_map
        if kwargs[k].frozen_params is not None:
            frozen_params[k] = kwargs[k].frozen_params
    if not frozen_params:
        frozen_params = None

    return CommonInit(frozen_params, params, scan_map, es_map)

def scan_init(model_cls, key, n_layer, *args, **kwargs):
    all_keys = jax.random.split(key, n_layer)
    all_params = []
    for t in range(n_layer):
        layer_vals = model_cls.rand_init(all_keys[t], *args, **kwargs)
        frozen_params_i, params_i, scan_map_i, es_map_i = layer_vals
        all_params.append(params_i)
        if t == 0:
            frozen_params = frozen_params_i
            scan_map = scan_map_i
            es_map = es_map_i
    params = jax.tree.map(lambda *x: jnp.stack(x), *all_params)
    scan_map = jax.tree.map(lambda x: (0,) + tuple(a + 1 for a in x), scan_map, is_leaf=lambda x: isinstance(x, tuple))
    return CommonInit(frozen_params, params, scan_map, es_map)

def merge_frozen(common, **kwargs):
    new_frozen_params = common.frozen_params or {}
    new_frozen_params = new_frozen_params | kwargs
    return common._replace(frozen_params=new_frozen_params)

def call_submodule(cls, name, common_params, *args, **kwargs):
    sub_common_params = common_params._replace(
        frozen_params=common_params.frozen_params[name] if common_params.frozen_params and name in common_params.frozen_params else None,
        params=common_params.params[name],
        es_tree_key=common_params.es_tree_key[name]
    )
    return cls._forward(sub_common_params, *args, **kwargs)

class Model:
    @classmethod
    def rand_init(cls, key, *args, **kwargs):
        """
        Initialize model

        returns frozen_params, params, scan_map, es_map as CommonInit
        """
        raise NotImplementedError("Randomize Weights is not implemented")

    @classmethod
    def forward(cls,
                noiser, frozen_noiser_params, noiser_params,
                frozen_params, params, es_tree_key, iterinfo, *args, **kwargs):
        """
        Forward pass of model

        returns just the output
        """
        return cls._forward(CommonParams(noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo), *args, **kwargs)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented")


class Parameter(Model):
    @classmethod
    def rand_init(cls, key, shape, scale, raw_value, dtype, *args, **kwargs):
        if raw_value is not None:
            params = raw_value.astype(dtype=dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype=dtype)
        
        frozen_params = None
        scan_map = ()
        es_map = PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        return common_params.noiser.get_noisy_standard(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo)

def sigmoid(x):
    return x

def tanh(x):
    return x

def clipped_add(*a):
    return jnp.clip(sum(x.astype(jnp.int32) for x in a), -MAX, MAX).astype(DTYPE)

class Embedding(Model):
    @classmethod
    def rand_init(cls, key, vocab_size, hidden_dim, dtype, *args, **kwargs):
        params = jnp.round(jax.random.normal(key, (vocab_size, hidden_dim)) * (2 ** FIXED_POINT)).astype(dtype)
        frozen_params = None
        scan_map = ()
        es_map = EMB_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_emb(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class MM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        params = jnp.round(jax.random.normal(key, (out_dim, in_dim)) * (2 ** FIXED_POINT)).astype(dtype)
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_mm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)


class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, use_bias, dtype, *args, **kwargs):
        if use_bias:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
                bias=Parameter.rand_init(key, None, None, jnp.zeros(out_dim, dtype=dtype), dtype)
            )
        else:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
            )

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        ans = call_submodule(MM, 'weight', common_params, x)
        if "bias" in common_params.params:
            ans += call_submodule(Parameter, 'bias', common_params)
        return ans

class MLP(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dims, use_bias, activation, dtype, *args, **kwargs):
        input_dims = [in_dim] + list(hidden_dims)
        output_dims = list(hidden_dims) + [out_dim]

        all_keys = jax.random.split(key, len(input_dims))

        merged_params = merge_inits(**{str(t): Linear.rand_init(all_keys[t], input_dims[t], output_dims[t], use_bias, dtype) for t in range(len(input_dims))})
        return merge_frozen(merged_params, activation=activation)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        num_blocks = len(common_params.params)
        for t in range(num_blocks):
            x = call_submodule(Linear, str(t), common_params, x)
            if t != num_blocks - 1:
                x = ACTIVATIONS[common_params.frozen_params['activation']](x)
        return x

class EGG_LN(Model):
    @classmethod
    def rand_init(cls, key, hidden_dim, dtype, *args, **kwargs):
        return merge_inits(
            weight=Parameter.rand_init(None, None, None, jnp.ones(hidden_dim, dtype=dtype) * (2 ** FIXED_POINT), dtype),
        )

    @classmethod
    def _forward(cls, common_params, x):
        # input is -127 to 127 range
        weight = call_submodule(Parameter, 'weight', common_params).astype(jnp.int32)
        abs_sum = (jnp.clip(jnp.dot(jnp.abs(x), jnp.ones_like(x), preferred_element_type=jnp.int32), min=1) // x.size)  # dividing by constant -> bit shift
        numerator = (x * weight).astype(jnp.int16).view(jnp.uint16)
        return common_params.noiser_params["DIVISION"][abs_sum][numerator]
    
    
class EGG_GRU(Model):
    @classmethod
    def rand_init(cls, key, hidden_dim, dtype, *args, **kwargs):
        # Wf, Uf, bf, Wh, Uh, bh
        keys = jax.random.split(key, 4)
        return merge_inits(
            Wf=MM.rand_init(keys[0], hidden_dim, hidden_dim, dtype),
            Uf=MM.rand_init(keys[1], hidden_dim, hidden_dim, dtype),
            bf=Parameter.rand_init(None, None, None, jnp.zeros(hidden_dim, dtype=dtype), dtype),
            Wh=MM.rand_init(keys[2], hidden_dim, hidden_dim, dtype),
            Uh=MM.rand_init(keys[3], hidden_dim, hidden_dim, dtype),
            bh=Parameter.rand_init(None, None, None, jnp.zeros(hidden_dim, dtype=dtype), dtype),
        )
    
    @classmethod
    def _forward(cls, common_params, x, state, *args, **kwargs):
        # x is standard fixed point (16x true value)
        # state is using full dynamic range (127x true value)
        ft = sigmoid(clipped_add(
            call_submodule(MM, 'Wf', common_params, x),
            call_submodule(MM, 'Uf', common_params, state),
            call_submodule(Parameter, 'bf', common_params)
        )) # scaled from -127 to 127

        # [0, 254] * [-127, 127] // 254 -> [-127, 127]
        gated_past = ((ft.astype(jnp.int32) + MAX) * state.astype(jnp.int32) >> (LOGMAX + 1)).astype(x.dtype)
        
        ht = tanh(clipped_add(
            call_submodule(MM, 'Wh', common_params, x),
            call_submodule(MM, 'Uh', common_params, gated_past),
            call_submodule(Parameter, 'bh', common_params)
        ))

        ht = state + (((ft.astype(jnp.int32) + MAX)) * (ht.astype(jnp.int32) - state.astype(jnp.int32)) >> (LOGMAX + 1)).astype(x.dtype)
        return ht, ht

class LayerEGG(Model):
    @classmethod
    def rand_init(cls, key, hidden_dim, dtype, *args, **kwargs):
        att_key, mlp_key = jax.random.split(key)
        return merge_inits(
            ln1=EGG_LN.rand_init(None, hidden_dim, dtype),
            att=EGG_GRU.rand_init(att_key, hidden_dim, dtype),
            ln2=EGG_LN.rand_init(None, hidden_dim, dtype),
            mlp=MLP.rand_init(mlp_key, hidden_dim, hidden_dim, [hidden_dim * 4], False, "relu", dtype)
        )

    @classmethod
    def _forward(cls, common_params, x, state, *args, **kwargs):
        residual = x
        x = call_submodule(EGG_LN, 'ln1', common_params, x)
        x, state = call_submodule(EGG_GRU, 'att', common_params, x, state)
        x = clipped_add(x, residual)

        residual = x
        x = call_submodule(EGG_LN, 'ln2', common_params, x)
        x = call_submodule(MLP, 'mlp', common_params, x, state)
        x = clipped_add(x, residual)
        return x, state

class EGG(Model):
    @classmethod
    def rand_init(cls, key, vocab_size, n_layer, hidden_dim, dtype, *args, **kwargs):
        key_emb, key_block, key_head = jax.random.split(key, 3)
        return merge_inits(
            emb=Embedding.rand_init(key_emb, vocab_size, hidden_dim, dtype),
            blocks=scan_init(LayerEGG, key_block, n_layer, hidden_dim, dtype),
            ln_out=EGG_LN.rand_init(None, hidden_dim, dtype),
            head=MM.rand_init(key_head, hidden_dim, vocab_size, dtype)
        )

    @classmethod
    def default_state(cls, params, config):
        return jnp.zeros_like(params["blocks"]["att"]["bh"])

    @classmethod
    def _forward(cls, common_params, x, state, *args, **kwargs):
        x = call_submodule(Embedding, 'emb', common_params, x)
        def block_loop(x, inputs):
            params_i, es_tree_key_i, state_i = inputs
            block_i = common_params._replace(frozen_params=common_params.frozen_params['blocks'], params=params_i, es_tree_key=es_tree_key_i)
            x, state_i = LayerEGG._forward(block_i, x, state_i)
            return x, state_i

        x, state = jax.lax.scan(block_loop, x, (common_params.params['blocks'], common_params.es_tree_key['blocks'], state), unroll=True)
        
        return call_submodule(MM, 'head', common_params, call_submodule(EGG_LN, 'ln_out', common_params, x)), state


### END MODEL IMPLEMENTATION

### Q EGGROLL IMPLEMENTATION

def fold_in(base_key_int32, new_int32):
    x = new_int32
    # rougly based on https://stackoverflow.com/a/12996028
    x = ((x >> 16) ^ x) * 0x45d9f3b
    return base_key_int32[0] ^ x

def get_common_start_idx(frozen_noiser_params, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"] # just shifting if noise_reuse is pow of 2

    true_thread_idx = (thread_id >> 1)

    actual_key = fold_in(jax.random.key_data(jax.random.fold_in(key, true_epoch)), true_thread_idx)
    start_idx = actual_key & (2**30 - 1)
    return start_idx, jnp.where(thread_id % 2 == 0, 1, -1)
    
def get_lora_update_params(BIG_RAND_MATRIX, frozen_noiser_params, iterinfo, param, key):
    a, b = param.shape
    r = frozen_noiser_params["rank"]

    start_idx, anti_sign = get_common_start_idx(frozen_noiser_params, iterinfo, param, key)
    lora_params = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, start_idx, (a+b)*r).reshape((a+b, r))
    
    B = lora_params[:b] # b x r
    A = lora_params[b:] # a x r

    return A * anti_sign, B

def get_nonlora_update_params(BIG_RAND_MATRIX, frozen_noiser_params, iterinfo, param, key):
    start_idx, anti_sign = get_common_start_idx(frozen_noiser_params, iterinfo, param, key)
    updates = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, start_idx, param.size * 2).reshape(param.shape + (2,)).astype(jnp.int32)
    return jnp.prod(updates, axis=-1) * anti_sign

def _simple_full_update(frozen_noiser_params, noiser_params, param, key, scores, iterinfo):
    updates = jax.vmap(partial(get_nonlora_update_params, noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params), in_axes=(0, None, None))(iterinfo, param, key)

    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    if frozen_noiser_params["use_clt"]:
        A = broadcasted_scores * updates.astype(jnp.int32)
    else:
        A = broadcasted_scores * jnp.sign(updates).astype(jnp.int32)
    Z = jnp.sum(A, axis=0)
    param_int32 = param.astype(jnp.int32)
    if frozen_noiser_params["fast_fitness"]:
        return jnp.clip(jnp.where(jnp.abs(Z) * (2 ** FIXED_POINT) < noiser_params["update_threshold"] * jnp.sqrt(scores.size) * (4 ** FIXED_POINT if frozen_noiser_params["use_clt"] else 1), param_int32, jnp.where(Z > 0, param_int32 + 1, param_int32 - 1)), -MAX, MAX).astype(param.dtype)
    else:
        return jnp.clip(jnp.where(jnp.abs(Z) < noiser_params["update_threshold"] * jnp.sqrt(scores.size) * (4 ** FIXED_POINT if frozen_noiser_params["use_clt"] else 1), param_int32, jnp.where(Z > 0, param_int32 + 1, param_int32 - 1)), -MAX, MAX).astype(dt)

def _simple_lora_update(frozen_noiser_params, noiser_params, param, key, scores, iterinfo):
    A, B = jax.vmap(partial(get_lora_update_params, noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params), in_axes=(0, None, None))(iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,1))
        
    if frozen_noiser_params["use_clt"]:
        if frozen_noiser_params["fast_fitness"]:
            A = broadcasted_scores * A
        else:
            A = broadcasted_scores.astype(jnp.int32) * A
            A = (A >> FIXED_POINT).astype(jnp.int8)
    else:
        A = broadcasted_scores * jnp.sign(A)
        B = jnp.sign(B)
    Z = jnp.einsum('nir,njr->ij', A, B, preferred_element_type=jnp.int32) # TODO: fix for rank > 1
    param_int32 = param.astype(jnp.int32)
    if frozen_noiser_params["fast_fitness"]:
        return jnp.clip(jnp.where(jnp.abs(Z) * (2 ** FIXED_POINT) < noiser_params["update_threshold"] * jnp.sqrt(scores.size) * (4 ** FIXED_POINT if frozen_noiser_params["use_clt"] else 1), param_int32, jnp.where(Z > 0, param_int32 + 1, param_int32 - 1)), -MAX, MAX).astype(param.dtype)
    else:
        return jnp.clip(jnp.where(jnp.abs(Z) < noiser_params["update_threshold"] * jnp.sqrt(scores.size) * (2 ** FIXED_POINT if frozen_noiser_params["use_clt"] else 1), param_int32, jnp.where(Z > 0, param_int32 + 1, param_int32 - 1)), -MAX, MAX).astype(param.dtype)

def _noop_update(noiser_params, base_sigma, ppf, param, key, scores, iterinfo, frozen_noiser_params):
    return param

class QEggRoll:
    @classmethod
    def init_noiser(cls, params, sigma_shift, update_threshold, *args, dtype='int8', noise_seed=0, noise_reuse=1, rank=1, use_clt=False, fast_fitness=True, **kwargs):
        """
        Return frozen_noiser_params and noiser_params
        """
        frozen_division = jnp.clip(jnp.arange(2**16).astype(jnp.int16)[None, :] // jnp.arange(2**8).astype(jnp.uint8)[:, None], -MAX, MAX).astype(jnp.int8)  # precalculating
        return {"noise_reuse": noise_reuse, "rank": rank, "use_clt": use_clt, "fast_fitness": fast_fitness}, {"BIG_RAND_MATRIX": (jax.random.normal(jax.random.key(noise_seed), 2**30) * (2 ** FIXED_POINT)).astype(dtype), "sigma_shift": sigma_shift, "update_threshold": update_threshold, "DIVISION": frozen_division}

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        # x is standard fixed point (16x true value)
        # param is also 16x true value and need to divide by sqrt(input size)
        base_ans = jnp.dot(x, param.T, preferred_element_type=jnp.int32)
        if iterinfo is not None:
            A, B = get_lora_update_params(noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params, iterinfo, param, base_key)
            perturbation = jnp.dot(x, B, preferred_element_type=jnp.int32) @ A.T.astype(jnp.int32)
            base_ans += perturbation >> (FIXED_POINT + noiser_params["sigma_shift"])
        return jnp.clip(base_ans // ((2 ** FIXED_POINT) * int(np.sqrt(param.shape[-1]))), -MAX, MAX).astype(param.dtype) # Just shifting if param is pow of 4

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        raise NotImplementedError("Tmm is not implemented")

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = param[x]
        if iterinfo is None:
            return base_ans
        base_ans = base_ans.astype(jnp.int32)
        A, B = get_lora_update_params(noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params, iterinfo, param, base_key)
        perturbation = jnp.dot(A[x], B.T, preferred_element_type=jnp.int32)
        base_ans += perturbation >> (FIXED_POINT + noiser_params["sigma_shift"])
        return jnp.clip(base_ans, -MAX, MAX).astype(param.dtype)

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None:
            return param
        base_ans = param.astype(jnp.int32)
        perturbation = get_nonlora_update_params(noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params, iterinfo, param, base_key)
        return jnp.clip(base_ans + (perturbation >> (FIXED_POINT + noiser_params["sigma_shift"])), -MAX, MAX).astype(param.dtype)

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        # NOTE: only half of the raw_scores due to antithetical sampling
        paired_scores = jnp.reshape(raw_scores, (-1, 2))
        if frozen_noiser_params["fast_fitness"]:
            return jnp.sign(paired_scores[:, 0] - paired_scores[:, 1]).astype(DTYPE)
        perf_diff = (paired_scores[:, 0] - paired_scores[:, 1]) * (2 ** FIXED_POINT)
        rms = jnp.sqrt(perf_diff ** 2).astype(jnp.int32)
        return (perf_diff * (2 ** FIXED_POINT) // rms).astype(DTYPE) # has to do slow int division, but only if not fast_fitness
        # return jnp.sign(paired_scores[:, 0] - paired_scores[:, 1]).astype(jnp.int8)

    @classmethod
    def _do_update(cls, frozen_noiser_params, noiser_params, param, base_key, fitnesses, iterinfos, map_classification):
        update_fn = [_simple_full_update, _simple_lora_update, _simple_lora_update, _noop_update][map_classification]

        if len(base_key.shape) == 0:
            updated_param = update_fn(frozen_noiser_params, noiser_params, param, base_key, fitnesses, iterinfos)
        else:
            # updated_param = jax.lax.scan(lambda _, x: (0, update_fn(frozen_noiser_params, noiser_params, x[0], x[1], fitnesses, iterinfos)), 0, xs=(param, base_key), unroll=True)[1]
            updated_param = jax.vmap(update_fn, in_axes=(None, None, 0, 0, None, None))(frozen_noiser_params, noiser_params, param, base_key, fitnesses, iterinfos)

        return updated_param
        
    
    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        iterinfos = jax.tree.map(lambda x: x[::2], iterinfos)
        return noiser_params, jax.tree.map(lambda p, k, m: cls._do_update(frozen_noiser_params, noiser_params, p, k, fitnesses, iterinfos, m), params, base_keys, es_map)

### END Q EGGROLL IMPLEMENTATION
FBIT = 4
EXP2TABLE = jnp.array((np.exp2(np.arange(256) / (2 ** FBIT)) * (2 ** FBIT)).astype(np.int32)) # can be precalculated
NOISER = QEggRoll

def get_int_ll(LOG2TABLE, logits, target):
    # log(2^{logits[target]} / sum(2^{logits})) = logits[target] - log(sum(2^logits))
    logits = logits.astype(jnp.int32) + 128 # [0, 255]
    target_logits = logits[target]
    logsumexp = LOG2TABLE[jnp.sum(EXP2TABLE[logits]) - 1]
    # Alternative: possibly implement fixed point log2? https://github.com/dmoulding/log2fix/blob/master/log2fix.c
    return target_logits - logsumexp

def generate_thread(frozen_noiser_params, frozen_params, es_tree_key, LOG2TABLE, noiser_params, params, input_tokens, target_tokens, state, thread_idx, epoch_num):
    iterinfo = None if epoch_num is None else (epoch_num, thread_idx)
    def inner_scan(state, inputs):
        input_tok, target_tok = inputs
        state = jax.lax.select(input_tok == 0, jnp.zeros_like(state), state)
        x, state = EGG.forward(NOISER, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo, input_tok, state)
        loss = get_int_ll(LOG2TABLE, x, target_tok)
        return state, loss
    state, losses = jax.lax.scan(inner_scan, state, (input_tokens, target_tokens))
    return jnp.sum(losses), state

def run_evolution():
    master_key = jax.random.key(args.seed)
    base_model_key = jax.random.fold_in(master_key, 0)
    base_es_key = jax.random.fold_in(master_key, 1)
    base_data_key = jax.random.fold_in(master_key, 2)

    cached_name = f"{args.seed}_{args.vocab_size}_{args.n_layer}_{args.n_embd}_{args.dtype}.model"
    path = Path(os.path.join(args.dir_path, cached_name))
    if path.is_file() and not args.regenerate_model:
        frozen_params, params, scan_map, es_map = load(path)
    else:
        print("REBUILDING PARAMS")
        full_params = EGG.rand_init(base_model_key, args.vocab_size, args.n_layer, args.n_embd, args.dtype)
        save(full_params, path, True)
        frozen_params, params, scan_map, es_map = full_params
    es_tree_key = simple_es_tree_key(params, base_es_key, scan_map)
    print("Num parameters", jax.tree.reduce(operator.add, jax.tree.map(lambda x: x.size, params)))


    if args.alpha > 1.0:
        all_alphas = 1.0 / ((np.exp2(args.alpha) - 1.0) * (np.arange(args.num_epochs) / args.num_epochs) + 1.0) # ON CPU, can be precalcualted
    else:
        all_alphas = args.alpha * np.ones(args.num_epochs)
    all_thresholds = (scipy.stats.norm.ppf(1-all_alphas / 2) * (2 ** FBIT)).astype(np.int32) # ON CPU, can be precalculated
    
    # print("Update threshold is", update_threshold / (2 ** FBIT), " instead of ", jax.scipy.stats.norm.ppf(1-args.alpha/2))

    frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma_shift, all_thresholds[0], dtype=args.dtype, noise_seed=args.seed, noise_reuse=args.noise_reuse, use_clt=args.use_clt, fast_fitness=args.fast_fitness)

    all_thread_idxes = jnp.arange(args.parallel_generations_per_gpu)
    if args.num_perturbations != 0:
        all_thread_idxes = all_thread_idxes % args.num_perturbations
    states = jnp.repeat(EGG.default_state(params, frozen_params)[None], args.parallel_generations_per_gpu, axis=0)

    LOG2TABLE = jnp.array((np.log2((np.arange(2**28) + 1) / (2 ** FBIT)) * (2 ** FBIT)).astype(np.int32)) # can be precalculated
    
    print("Compiling generate batch")
    start_time = time.time()
    v_generate_thread = jax.jit(jax.vmap(partial(generate_thread, frozen_noiser_params, frozen_params, es_tree_key), in_axes=(None, None, None, 0, 0, 0, 0, None)), donate_argnums=5).lower(
        LOG2TABLE, noiser_params, params, jax.ShapeDtypeStruct((args.parallel_generations_per_gpu, args.tokens_per_update), jnp.dtype('uint8')), jax.ShapeDtypeStruct((args.parallel_generations_per_gpu, args.tokens_per_update), jnp.dtype('uint8')), states, all_thread_idxes, 0
    ).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(v_generate_thread.memory_analysis())

    
    print("Compiling validate")
    validation_dataset = np.load(os.path.join(args.dir_path, args.valid_output_path))
    num_tokens_per_validation_thread = validation_dataset.size // args.validation_batch_size
    validation_dataset = validation_dataset[:args.validation_batch_size*num_tokens_per_validation_thread].reshape((args.validation_batch_size, -1))
    start_time = time.time()
    validate_model = jax.jit(jax.vmap(partial(generate_thread, frozen_noiser_params, frozen_params, es_tree_key), in_axes=(None, None, None, 0, 0, 0, 0, None)), donate_argnums=5).lower(
        LOG2TABLE, noiser_params, params, validation_dataset[:, :-1], validation_dataset[:, 1:], jnp.repeat(EGG.default_state(params, frozen_params)[None], args.validation_batch_size, axis=0), jnp.arange(args.validation_batch_size), None
    ).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(validate_model.memory_analysis())
    

    print()
    print("Compiling do update")
    start_time = time.time()
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map)).lower(
        noiser_params, params, jax.ShapeDtypeStruct((args.parallel_generations_per_gpu >> 1,), jnp.dtype('int8')), (jnp.zeros(args.parallel_generations_per_gpu, dtype=jnp.int32), all_thread_idxes)
    ).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(jit_update.memory_analysis())

    full_dataset = np.load(os.path.join(args.dir_path, args.train_output_path))
    num_sequences = args.parallel_generations_per_gpu // args.group_size
    segments_per_sequence = (full_dataset.size - num_sequences) // (args.tokens_per_update * num_sequences)
    tokens_per_sequence = segments_per_sequence * args.tokens_per_update + 1
    print("total number of sequences is", num_sequences)
    print("tokens per sequence is", tokens_per_sequence)
    print("number of segments per sequence (total epochs until resample)", segments_per_sequence)
    print("TARGETS: unigram gets 5.0 bits, bigram gets 4.0, and gzip w/ max compression gives 2.767 bits")

    truncated_dataset = full_dataset[:num_sequences * tokens_per_sequence].reshape((num_sequences, tokens_per_sequence))

    # full_name = f"int8_a{update_threshold}_s{args.sigma_shift}_{args.parallel_generations_per_gpu}x{args.tokens_per_update}x{args.noise_reuse}"
    full_name = f"{args.dtype}_{args.n_embd}D{args.n_layer}L_a{args.alpha}_s{args.sigma_shift}_{args.parallel_generations_per_gpu}x{args.tokens_per_update}x{args.noise_reuse}"
    print("Run name", full_name)
    if args.track:
        run = wandb.init(
            project=args.wandb_project,
            config=args,
            name=full_name
        )

    for epoch in tqdm.trange(args.num_epochs):
        full_start_time = time.time()

        noiser_params["update_threshold"] = all_thresholds[epoch]

        iterinfo = (jnp.full(args.parallel_generations_per_gpu, epoch, dtype=jnp.int32), all_thread_idxes)

        start_tok = (epoch % segments_per_sequence) * args.tokens_per_update
        full_obs = truncated_dataset[:, start_tok:start_tok+args.tokens_per_update + 1]
        batch_obs = jnp.repeat(full_obs, args.group_size, axis=0)

        batch_inputs = batch_obs[:, :-1]
        batch_targets = batch_obs[:, 1:]

        start_time = time.time()
        raw_scores, states = jax.block_until_ready(v_generate_thread(LOG2TABLE, noiser_params, params, batch_inputs, batch_targets, states, jnp.arange(args.parallel_generations_per_gpu), epoch))
        generate_time = time.time() - start_time
        # num_nans = jnp.sum(jnp.isnan(raw_scores))

        start_time = time.time()
        raw_scores = jnp.where(jnp.isnan(raw_scores), jnp.nanmin(raw_scores) - 100, raw_scores)
        float_scores = raw_scores / (args.tokens_per_update * (2**FBIT))  # only used for logging
        avg_fitness = jnp.mean(float_scores)
        min_fitness = jnp.min(float_scores)
        max_fitness = jnp.max(float_scores)

        fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        noiser_params, new_params = jax.block_until_ready(jit_update(noiser_params, params, fitnesses, iterinfo))
        update_time = time.time() - start_time

        parameter_differences = jax.tree.map(lambda x, y: jnp.mean(x != y), params, new_params)
        lora_differences = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == 1 else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == 1 else 0.0, es_map))  # only used for logging
        nonlora_differences = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == 0 else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == 0 else 0.0, es_map))  # only used for logging

        params = new_params
        end_time = time.time()

        stats = {
            "avg_fitness": -avg_fitness,
            "min_fitness": -min_fitness,
            "max_fitness": -max_fitness,
            "lora_differences": lora_differences,
            "nonlora_differences": nonlora_differences,
            "throughput": args.tokens_per_update * args.parallel_generations_per_gpu / (end_time - full_start_time),
            "generate_time": generate_time,
            "update_time": update_time
        }

        if args.validate_every != 0 and epoch % args.validate_every == args.validate_every - 1:
            valid_start_time = time.time()
            validate_states = EGG.default_state(params, frozen_params)
            raw_scores, _ = jax.block_until_ready(validate_model(LOG2TABLE, noiser_params, params, validation_dataset[:, :-1], validation_dataset[:, 1:], jnp.repeat(EGG.default_state(params, frozen_params)[None], args.validation_batch_size, axis=0), jnp.arange(args.validation_batch_size), None))
            valid_time = time.time() - valid_start_time
            float_validation_score = -jnp.sum(raw_scores) / ((validation_dataset.size - args.validation_batch_size) * (2 ** FBIT))
            valid_throughput = (validation_dataset.size - args.validation_batch_size) / valid_time
            stats["validation_score"] = float_validation_score
            stats["valid_time"] = valid_time
            stats["validation_throughput"] = valid_throughput
            print("validation score", float_validation_score, "validation throughput is", valid_throughput, "validation time is", valid_time)
        
        if args.track:
            run.log(stats)
        elif epoch % 10 == 0:
            print(min_fitness, max_fitness)
            print(avg_fitness)
            print(f"\tlora changes: {lora_differences}; nonlora changes: {nonlora_differences}")
            print(f"\tmax lora: {jax.tree.reduce(max, jax.tree.map(lambda x, y: jnp.max(jnp.abs(x)) if y == 1 else 0.0, new_params, es_map))}; max nonlora: {jax.tree.reduce(max, jax.tree.map(lambda x, y: jnp.max(jnp.abs(x)) if y == 0 else 0.0, new_params, es_map))}")
            print(f"\tgenerate time: {generate_time}; update time: {update_time}; throughput: {stats['throughput']}")
            # print("max fitness value", jnp.max(jnp.abs(fitnesses)))


def build_dataset():
    path = Path(os.path.join(args.dir_path, args.train_output_path))
    if path.is_file():
        return
        
    os.makedirs(args.dir_path, exist_ok=True)
    
    print("getting dataset")
    ds = load_dataset("JeanKaddour/minipile")

    file_map = {
        'validation': args.valid_output_path,
        'test': args.test_output_path,
        'train': args.train_output_path,
    }
    file_map = {k: os.path.join(args.dir_path, v) for k, v in file_map.items()}

    for k in ds:
        print("loading", k)
        arrays = []
        for f in tqdm.tqdm(ds[k]['text']):
            arrays.append(np.array([0] + list(f.encode("utf-8")), dtype=np.uint8))
        print("generating numpy array")
        out_array = np.concatenate(arrays, dtype=np.uint8)
        print("shape is", out_array.shape)
        print("saving numpy array to", file_map[k])
        np.save(file_map[k], out_array)

if __name__ == "__main__":
    build_dataset()

    run_evolution()
