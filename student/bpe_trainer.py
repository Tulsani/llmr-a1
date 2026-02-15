from collections import Counter
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
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def merge_pair_in_counts(word_counts,pair, new_token):
    new_word_counts = Counter()
    for word, count in word_counts.items():
        # skip over words that don't contain the pair
        if pair[0] not in word or pair[1] not in word:
            new_word_counts[word] += count
            continue

        # replace all occurrences of the pair with the new token
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_counts[tuple(new_word)] += count
    return new_word_counts

def parellel_tokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)

    # decode the chunk data to a string for regex processing
    chunk_text = chunk_data.decode("utf-8")

    # remove special character, we split on them
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, chunk_text)
    else:
        parts = [chunk_text]
    
    word_counts = Counter()
    for part in parts:
        for match in re.finditer(PAT, part):
            word = match.group(0)
            #converting to bytes and add to counter
            b = word.encode("utf-8")
            symbols = tuple(bytes([x]) for x in b) 
            word_counts[symbols] += 1
    return word_counts

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    """
    returns vocab, merges
    """
    vocab = {}
    # setting up intiale vocab
    for i in range(0,256):
        vocab[i] = bytes([i])

    # adding special tokens to vocab
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")
    
    # chunking - getting the boundaries
    num_processes = CONSTANT_NUM_PROCESS

    # should this be a function
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    chunk_args = [] # tuple of (input_path, start, end, special_tokens)

    for start, end in zip(boundaries[:-1], boundaries[1:]): 
        chunk_args.append((input_path, start, end, special_tokens))

    # parallel tokenization
    with Pool(num_processes) as pool:
        results = pool.map(parellel_tokenize_chunk, chunk_args) # call tokenize_chunk on each chunk in parallel
    
    # collecting count across word chunks
    word_counts = Counter()

    for chunk_count in results:
        word_counts.update(chunk_count)
    
    # unique words
    print(f"Unique pretokens: {len(word_counts)}")

    # merging
    merges = []

    base_vocab_size = len(vocab)
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # count all possible pairs
        pair_counts = Counter()
        for word, count in word_counts.items():
            #skip over word of size 2
            if len(word) < 2:
                continue

            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pair_counts[pair] += count

        if not pair_counts:
            print("No more pairs to merge.")
            break

        # find most common pair
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # add to merges
        merges.append(most_frequent_pair)

        # create new token for the merged pair
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[base_vocab_size] = new_token
        base_vocab_size+= 1

        word_counts = merge_pair_in_counts(word_counts, most_frequent_pair, new_token)

    print(f"BPE Complete Final vocab size: {len(vocab)}")

    
    return vocab, merges


def train_bpe_tinystories(vocab_size =10000,input_path= "./data/TinyStoriesV2-GPT4-train.txt"):
    # lets maangge via pickle
    print("Training BPE on TinyStoriesV2-GPT4-train.txt")
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open("./tinystories_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("./tinystories_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    print("Vocab and merges saved")

    # prechecks
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token length in vocab: {longest_token}")
    print(f"Decoded longest: {longest_token.decode('utf-8', errors='ignore')}")

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe_tinystories()