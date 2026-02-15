from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO
import os
import regex as re
import pickle


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CONSTANT_NUM_PROCESS = 4


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def parallel_tokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)

    chunk_text = chunk_data.decode("utf-8")

    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, chunk_text)
    else:
        parts = [chunk_text]

    word_counts = Counter()
    for part in parts:
        for match in re.finditer(PAT, part):
            word = match.group(0)
            b = word.encode("utf-8")
            symbols = tuple(bytes([x]) for x in b)
            word_counts[symbols] += 1
    return word_counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    vocab = {}
    # setting up intiale vocab
    for i in range(256):
        vocab[i] = bytes([i])
    # adding special tokens to vocab
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")

    # parallel tokenization
    num_processes = CONSTANT_NUM_PROCESS
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        results = pool.map(parallel_tokenize_chunk, chunk_args)

    word_counts = Counter()
    for chunk_count in results:
        word_counts.update(chunk_count)

    print(f"Unique pretokens: {len(word_counts)}")

    # pair counts
    pair_counts = defaultdict(int)
    for word, count in word_counts.items():
        for j in range(len(word) - 1):
            pair_counts[(word[j], word[j + 1])] += count

    # merge loop
    merges = []
    base_vocab_size = len(vocab)
    num_merges = vocab_size - base_vocab_size

    for merge_i in range(num_merges):
        if not pair_counts:
            print("No more pairs to merge.")
            break

        # find most frequent pair
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[base_vocab_size] = new_token
        base_vocab_size += 1

        if merge_i % 100 == 0:
            print(f"Merge {merge_i}/{num_merges}: count={pair_counts[best_pair]}")

        # Update word counts with the new merged token
        pair_freq_changes = defaultdict(int)
        new_word_counts = defaultdict(int)

        for word, count in word_counts.items():
            if len(word) < 2:
                new_word_counts[word] += count
                continue

            new_word = []
            i = 0
            found = False
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    found = True

                    # old pair to the left is destroyed
                    if new_word:
                        pair_freq_changes[(new_word[-1], best_pair[0])] -= count
                    # old pair to the right is destroyed
                    if i + 2 < len(word):
                        pair_freq_changes[(best_pair[1], word[i + 2])] -= count

                    new_word.append(new_token)

                    # new pair to the left is created
                    if len(new_word) > 1:
                        pair_freq_changes[(new_word[-2], new_token)] += count
                    # new pair to the right is created
                    if i + 2 < len(word):
                        pair_freq_changes[(new_token, word[i + 2])] += count

                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            if found:
                new_word_counts[tuple(new_word)] += count
            else:
                new_word_counts[word] += count

        word_counts = new_word_counts

        # Remove the merged pair from pair counts
        del pair_counts[best_pair]
        for pair, change in pair_freq_changes.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + change
            if pair_counts[pair] <= 0:
                if pair in pair_counts:
                    del pair_counts[pair]

    print(f"BPE Complete. Final vocab size: {len(vocab)}")
    return vocab, merges


def train_bpe_tinystories(vocab_size=10000, input_path="./data/TinyStoriesV2-GPT4-train.txt"):
    print("Training BPE on TinyStoriesV2-GPT4-train.txt")
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open("./tinystories_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("./tinystories_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print("Vocab and merges saved")

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token length in vocab: {longest_token}")
    print(f"Decoded longest: {longest_token.decode('utf-8', errors='ignore')}")

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe_tinystories()