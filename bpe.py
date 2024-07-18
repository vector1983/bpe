corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    # use gpt2 pre tokenization, we don't implement this part ourself.
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [ word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)


alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
print(alphabet)

# initialized vocab
vocab = ["<|endoftext|>"] + alphabet.copy()

splits = {word: [c for c in word] for word in word_freqs.keys()}
print(splits)

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

# compute the pair frequency
pair_freqs = compute_pair_freqs(splits)
print(pair_freqs)

# get the best pair
def get_best_pair(pair_freqs):
    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair

def merge_pair(a, b , splits):
    """
    :param a: pair element
    :param b: another pair element
    :param splits: { <word>: [w, o, r, d], ...}
    :return: new splits with pair replacement
    """

    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a+b] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits

vocab_size = 50
merges = {}

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = get_best_pair(pair_freqs)
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])


print(merges)
print(vocab)