import pickle
import regex as re 
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        #self.special_tokens = special_tokens

        self.byte_to_token_id = {v: k for k, v in vocab.items()}

        self.special_tokens = special_tokens or []


        # add special characters to vocab
        vocab_index = max(vocab.keys()) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_token_id:
                self.vocab[vocab_index] = token_bytes
                self.byte_to_token_id[token_bytes] = vocab_index
                vocab_index += 1
        
        # intialize priority of merges
        self.merge_priorities = {merge: i for i, merge in enumerate(merges)}

        #compile regex pattern for tokenization
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.token_pattern = re.compile(pattern)
        else:
            self.token_pattern = None

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        # load vocab and merges from files
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        # encode text into list of token ids
        if self.special_tokens:
            #split on special tokens
            self.special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            parts = re.split(f"({self.special_pattern})", text)
        
            token_ids = []
            for part in parts:
                if not part: # move over empty strings
                    continue

                if part in self.special_tokens:
                    # if the part is a special token, add its token id directly
                    token_ids.append(self.byte_to_token_id[part.encode("utf-8")])
                else:
                    # text we apply BPE
                    token_ids.extend(self.encode_part(part))
            
            return token_ids
        else:
            # no special tokens, just apply BPE to the whole text
            return self.encode_part(text)
            

    def encode_iterable(self,iterable):
        # encode an iterable of text into list of token ids
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self,ids):
        # decode list of token ids back into text
        btype_seq = b""
        for token_id in ids:
            if token_id in self.vocab:
                btype_seq += self.vocab[token_id]
            else:
                print(f"Warning: token id {token_id} not in vocab, skipping")

        return btype_seq.decode("utf-8", errors="ignore")

    #internal methods
    def encode_part(self, text):
        # encode a part of text (without special tokens) using BPE merges
        # start with byte-level tokenization
        token_ids = []
        for match in re.finditer(PAT, text):
            word = match.group(0)
            #converting to bytes and add to counter
            b = word.encode("utf-8")
            #tuple of byte tokens
            word_tokens = tuple(bytes([x]) for x in b)

            # apply merges to the word tokens
            merged_tokens = self.apply_merges(word_tokens)

            # convert tokens to ids
            for token in merged_tokens:
                if token in self.byte_to_token_id:
                    token_ids.append(self.byte_to_token_id[token])
                else:
                    for bb in token: 
                        token_ids.append(self.byte_to_token_id[bytes([bb])])

        return token_ids
    
    def apply_merges(self, word):
        while True:
            # find all pairs
            pairs = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_priorities:
                    pairs.append((self.merge_priorities[pair], i, pair))
            
            # stopping case
            if not pairs:
                break

            # find highest priority pair
            pairs.sort()  # sort by priority
            _, index, pair_to_merge = pairs[0]

            # merge the pair
            new_word = []
            j=0
            while j < len(word):
                if  j == index:
                    new_word.append(pair_to_merge[0] + pair_to_merge[1])
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            
            word = tuple(new_word)

        return list(word)