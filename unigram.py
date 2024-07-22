corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    # use gpt2 pre tokenization, we don't implement this part ourself.
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [ word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

char_freqs = defaultdict(int)
subword_freqs = defaultdict(int)

for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        for j in range(i+2, len(word) + 1):
            subword_freqs[word[i:j]] += freq

sorted_subwords = sorted(subword_freqs.items(), key=lambda x:x[1], reverse=True)
#print(len(sorted_subwords))


# use char and best subwords to create intial vocab(sentencePiece uses enhanced suffix array(ESA) algorithm, it is a more efficient one)
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}
print(token_freqs)

from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq/total_sum) for token, freq in token_freqs.items()}

# viterbi algorithm
# model key token: status， 所有可能的token是状态
# word: observations， 目前要tokenization的词是观测集合
# model[token]: 可以得到不同状态（token）转换的概率，状态转移概率
# 目地：找到一个最有可能的token序列，使得观测概率最大

# according model, 计算word最大概率的tokens组合,word每个位置组成的最大概率的token
def encode_word(word, model):
    best_segmentations = [{"start" : 0, "score": 1}] + [{"start":None, "score": None} for _ in range(len(word))]
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx+1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # if we have found a better segmentation ending at end_idx, we update
                # score越小越好，越小概率越大
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

print(encode_word("Hopefully", model))
print(encode_word("This", model))

# 整个model的loss等于model中所有word的loss相加，word的loss为上面的score。
# 计算整个corpus的loss
def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss


    return loss

compute_loss(model)
import copy

# 获取model删除某个token后loss增加的值
def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores


scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])


# 删除那些对loss增加小的，不要删除造成loss增加大的
percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_score = sorted(scores.items(), key=lambda x : x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model)*percent_to_remove)):
        _ = token_freqs.pop(sorted_score[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq/total_sum) for token, freq in token_freqs.items()}

def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encode_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encode_words, [])

