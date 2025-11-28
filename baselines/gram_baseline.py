import os
import numpy as np

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cached_files")
# valid_output_path = "minipile_train.npy"
valid_output_path = "minipile_valid.npy"

full_dataset = np.load(os.path.join(dir_path, valid_output_path))[:10000000]
print(full_dataset.size)

first_values = full_dataset[:-1]
second_values = full_dataset[1:]

print(np.unique_counts(full_dataset)[0].size)
counts = np.zeros(256, dtype=int)
unique_counts = np.unique_counts(full_dataset)
counts[unique_counts.values] = unique_counts.counts

# print(counts)

bigram_counts = np.zeros((256, 256), dtype=int)
for i in range(256):
    valid_indices = np.nonzero(first_values == i)
    unique_counts = np.unique_counts(second_values[valid_indices])
    bigram_counts[i][unique_counts.values] = unique_counts.counts

unigram_prob = counts / counts.sum()

print("unigram entropy is", -np.sum(unigram_prob * np.log2(np.maximum(unigram_prob, 1e-6))))

bigram_prob = bigram_counts / np.maximum(bigram_counts.sum(axis=-1, keepdims=True), 1)
individual_entropies = -np.sum(bigram_prob * np.log2(np.maximum(bigram_prob, 1e-6)), axis=-1)
print("bigram entropy is", np.sum(unigram_prob * individual_entropies))
