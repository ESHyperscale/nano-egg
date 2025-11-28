import jax
import jax.numpy as jnp

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "cache/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import optax

from functools import partial

import tyro
from dataclasses import dataclass

from tqdm import trange
import time

import numpy as np

import wandb

@dataclass
class Args:
    seed: int = 0

    batch_size: int = 512
    num_epochs: int = 1000

    n_layer: int = 6
    n_embd: int = 256
    vocab_size: int = 256

    lr: float = 0.15
    lr_decay: bool = False

    tokens_per_update: int = 100

    dir_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../cached_files")
    train_output_path: str = "minipile_train.npy"
    valid_output_path: str = "minipile_valid.npy"
    test_output_path: str = "minipile_test.npy"

    wandb_project: str = "HyperscalePretrainingBaseline"
    tag: str = ""
    track: bool = False

    validate_every: int = 10
    validation_batch_size: int = 1024

args = tyro.cli(Args)

def xavier_uniform(key, shape, dtype):
    scale = jnp.sqrt(6/(shape[-1] + shape[-2]))
    return jax.random.uniform(key=key, shape=shape, minval=-scale, maxval=scale, dtype=dtype)

def rand_init(key, n_embd, n_layer, vocab_size, ctx_len, dtype=jnp.float32):
    _key1, _key2, _key3, _key4, _key5, _key6, _key7, _key8, _key9 = jax.random.split(key, 9)
    params = {}
    params['pos_embed'] = {'weight': xavier_uniform(_key1, (ctx_len, n_embd), dtype)}
    params['emb'] = {'weight': xavier_uniform(_key2, (vocab_size, n_embd), dtype)}
    params['head'] = {'weight': xavier_uniform(_key3, (vocab_size, n_embd), dtype), 'bias': jnp.zeros(vocab_size, dtype)}

    params['blocks'] = {
        'attn': {
            'query': {'weight': xavier_uniform(_key4, (n_layer, n_embd, n_embd), dtype)},
            'key': {'weight': xavier_uniform(_key5, (n_layer, n_embd, n_embd), dtype)},
            'value': {'weight': xavier_uniform(_key6, (n_layer, n_embd, n_embd), dtype)},
            'out': {'weight': xavier_uniform(_key7, (n_layer, n_embd, n_embd), dtype)}
        },
        'mlp': {
            'ff0': {'weight': xavier_uniform(_key8, (n_layer, 4 * n_embd, n_embd), dtype), 'bias': jnp.zeros((n_layer, 4 * n_embd), dtype)},
            'ff1': {'weight': xavier_uniform(_key9, (n_layer, n_embd, 4 * n_embd), dtype), 'bias': jnp.zeros((n_layer, n_embd), dtype)}
        },
        'ln1': {'bias': jnp.zeros((n_layer, n_embd), dtype), 'weight': jnp.ones((n_layer, n_embd), dtype)},
        'ln2': {'bias': jnp.zeros((n_layer, n_embd), dtype), 'weight': jnp.ones((n_layer, n_embd), dtype)}
    }

    params['ln_out'] = {'bias': jnp.zeros((n_embd), dtype), 'weight': jnp.ones((n_embd), dtype)}
    return params


def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']

def attention(x, params):
    T = x.shape[0]
    S = 64
    H = x.shape[1] // S
    q = jnp.reshape(x @ params['query']['weight'].T, (T, H, S))
    k = jnp.reshape(x @ params['key']['weight'].T, (T, H, S))
    v = jnp.reshape(x @ params['value']['weight'].T, (T, H, S))
    attn_output = jnp.reshape(jax.nn.dot_product_attention(q, k, v, is_causal=True), x.shape)
    return attn_output @ params['out']['weight'].T

def mlp_forward(x, params):
    x = x @ params['ff0']['weight'].T + params['ff0']['bias']
    x = jax.nn.gelu(x)
    x = x @ params['ff1']['weight'].T + params['ff1']['bias']
    return x

def forward(params, tokens):
    x = params['emb']['weight'][tokens] + params['pos_embed']['weight']
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def block_loop(x, block):
        x = x + attention(layer_norm(x, block['ln1']), block['attn'])
        x = x + jax.vmap(mlp_forward, in_axes=(0, None))(layer_norm(x, block['ln2']), block['mlp'])
        return x, 0
    x, _ = jax.lax.scan(block_loop, x, params['blocks'])
    return layer_norm(x, params['ln_out']) @ params['head']['weight'].T + params['head']['bias']

def loss(params, in_tokens, out_tokens):
    logits = forward(params, in_tokens)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, out_tokens) / jnp.log(2)) # convert to bits/byte

def batch_loss(params, in_tokens, out_tokens):
    return jnp.mean(jax.vmap(loss, in_axes=(None, 0, 0))(params, in_tokens, out_tokens))

fast_batch_grad = jax.value_and_grad(batch_loss)





params = rand_init(jax.random.key(args.seed), args.n_embd, args.n_layer, args.vocab_size, args.tokens_per_update)

full_dataset = np.load(os.path.join(args.dir_path, args.train_output_path))
num_sequences = args.batch_size
segments_per_sequence = (full_dataset.size - num_sequences) // (args.tokens_per_update * num_sequences)
tokens_per_sequence = segments_per_sequence * args.tokens_per_update + 1
print("total number of sequences is", num_sequences)
print("tokens per sequence is", tokens_per_sequence)
print("number of segments per sequence (total epochs until resample)", segments_per_sequence)
truncated_dataset = full_dataset[:num_sequences * tokens_per_sequence].reshape((num_sequences, tokens_per_sequence))

print("Number of parameters:", jax.tree.reduce(lambda *x: sum(x), jax.tree.map(jnp.size, params)) / 1000000, "million")
# solver = optax.adamw(LR)
# solver = optax.contrib.prodigy(1.0)
if args.lr_decay:
    solver = optax.sgd(optax.schedules.linear_schedule(args.lr, 0.0, args.num_epochs))
else:
    solver = optax.sgd(args.lr)
# solver = optax.sgd(args.lr, momentum=0.9)
# solver = optax.contrib.dowg()
# solver = optax.contrib.mechanize(optax.sgd(1.0))
optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))

def do_update(params, optimizer, input_tokens, output_tokens):
    loss, grad = fast_batch_grad(params, input_tokens, output_tokens)
    updates, optimizer = solver.update(grad, optimizer, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer, loss

print("Starting compilation")
start_time = time.time()
fast_update = jax.jit(do_update, donate_argnums=(0, 1)).lower(
    params, optimizer,
    jax.ShapeDtypeStruct((args.batch_size, args.tokens_per_update), jnp.dtype('uint8')),
    jax.ShapeDtypeStruct((args.batch_size, args.tokens_per_update), jnp.dtype('uint8'))
).compile()
print("Compilation takes", time.time() - start_time)
print(f"Memory usage with {args.batch_size}x{args.tokens_per_update}")
print(fast_update.memory_analysis())

print("Compiling validate")
validation_dataset = np.load(os.path.join(args.dir_path, args.valid_output_path))
num_validation_sequences = args.validation_batch_size
validation_segments_per_sequence = (validation_dataset.size - num_validation_sequences) // (args.tokens_per_update * num_validation_sequences)
tokens_per_validation_sequence = validation_segments_per_sequence * args.tokens_per_update + 1
truncated_validation_dataset = validation_dataset[:num_validation_sequences * tokens_per_validation_sequence].reshape((num_validation_sequences, tokens_per_validation_sequence))
validate_model = jax.jit(batch_loss).lower(
    params, jax.ShapeDtypeStruct((args.validation_batch_size, args.tokens_per_update), jnp.dtype('uint8')),
    jax.ShapeDtypeStruct((args.validation_batch_size, args.tokens_per_update), jnp.dtype('uint8'))
).compile()


full_name = f"tf_{args.n_embd}D{args.n_layer}L_{args.lr}" + ("decay" if args.lr_decay else "") + f"_{args.batch_size}x{args.tokens_per_update}"

print("Run name", full_name)
if args.track:
    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=full_name
    )

for epoch in trange(args.num_epochs):
    start_tok = (epoch % segments_per_sequence) * args.tokens_per_update
    full_obs = truncated_dataset[:, start_tok:start_tok+args.tokens_per_update + 1]
    params, optimizer, loss = fast_update(params, optimizer, full_obs[:, :-1], full_obs[:, 1:])

    stats = {
        "loss": loss,
        "data": (epoch + 1) * args.tokens_per_update * args.batch_size,
    }
    
    if args.validate_every != 0 and epoch % args.validate_every == args.validate_every - 1:
        validation_losses = []
        for t in trange(validation_segments_per_sequence):
            start_tok = t * args.tokens_per_update
            full_obs = truncated_validation_dataset[:, start_tok:start_tok+args.tokens_per_update + 1]
            validation_losses.append(validate_model(params, full_obs[:, :-1], full_obs[:, 1:]))
        validation_loss = np.mean(validation_losses)
        stats["validation_loss"] = validation_loss
        print("validation score", validation_loss)

    if args.track:
        run.log(stats)
    elif epoch % 10 == 0:
        print(f"iter {epoch}: {loss}")
