# Modified from https://github.com/ckiplab/ckiptagger/blob/50add4192087927342f0bc19137a169c608f8ebe/src/api.py

def construct_dictionary(word_to_weight):
    length_word_weight = {}

    for word, weight in word_to_weight.items():
        if not word: continue
        try:
            weight = float(weight)
        except ValueError:
            continue
        length = len(word)
        if length not in length_word_weight:
            length_word_weight[length] = {}
        length_word_weight[length][word] = weight

    length_word_weight = sorted(length_word_weight.items())

    return length_word_weight

def _get_forced_chunk_set(sentence, length_word_weight):

    chunk_to_weight = {}

    for i in range(len(sentence)):
        for length, word_to_weight in length_word_weight:
            word = sentence[i:i+length]
            if word in word_to_weight:
                chunk_to_weight[(i, i+length)] = word_to_weight[word]

    chunk_set = set()
    # empty_sentence = [True] * len(sentence)

    for (l, r), w in sorted(chunk_to_weight.items(), key=lambda x: (x[1], x[0][1]-x[0][0]), reverse=True):
        chunk_set.add((w, l, r))
        '''
        empty = True
        for i in range(l, r):
            if not empty_sentence[i]:
                empty = False
                break
        if not empty: continue

        chunk_set.add((w, l, r))
        for i in range(l, r):
            empty_sentence[i] = False
        '''

    return chunk_set

def _soft_force_seq_sentence(forced_chunk_set, seq_sentence):
    mask = [0.0] * len(seq_sentence)
    for w, l, r in sorted(forced_chunk_set, key=lambda x:x[0]):
        if max(mask[l:r]) > w: continue
        if seq_sentence[l] != "B": continue
        if r<len(seq_sentence) and seq_sentence[r] != "B": continue
        for i in range(l+1,r):
            seq_sentence[i] = "I"
        mask[l:r] = [w] * (r-l)
    return seq_sentence

def _hard_force_seq_sentence(forced_chunk_set, seq_sentence):
    mask = [0] * len(seq_sentence)
    for w, l, r in sorted(forced_chunk_set, key=lambda x:x[0]):
        if max(mask[l:r]) > w: continue
        seq_sentence[l] = "B"
        for i in range(l+1,r):
            seq_sentence[i] = "I"
        if r < len(seq_sentence):
            seq_sentence[r] = "B"
        mask[l:r] = [w] * (r-l)
    return seq_sentence

def _get_word_sentence_from_seq_sentence(sentence, seq_sentence):
    assert len(sentence) == len(seq_sentence)
    if not sentence:
        return []

    word_sentence = []
    word = sentence[0]
    for character, label in zip(sentence[1:], seq_sentence[1:]):
        if label == "B":
            word_sentence.append(word)
            word = ""
        word += character
    word_sentence.append(word)

    return word_sentence

def _run_word_segmentation_with_dictionary(word_sentence, recommend_dictionary=None, coerce_dictionary=None):
    if recommend_dictionary is None and coerce_dictionary is None:
        assert ValueError('Either recommend_dictionary or coerce_dictionary should not be None')

    sentence = "".join(word_sentence)
    seq_sentence = []
    for word in word_sentence:
        seq_sentence.append("B")
        for character in word[1:]:
            seq_sentence.append("I")
    
    if recommend_dictionary is not None:
        recommend_chunk_set = _get_forced_chunk_set(sentence, recommend_dictionary)
        seq_sentence = _soft_force_seq_sentence(recommend_chunk_set, seq_sentence)
    if coerce_dictionary is not None:
        coerce_chunk_set = _get_forced_chunk_set(sentence, coerce_dictionary)
        seq_sentence = _hard_force_seq_sentence(coerce_chunk_set, seq_sentence)

    word_sentence = _get_word_sentence_from_seq_sentence(sentence, seq_sentence)
    return word_sentence

if __name__ == '__main__':
    dicti = [(2, {'公有': 2.0}), (3, {'土地公': 1.0, '土地婆': 1.0}), (5, {'緯來體育台': 1.0})]
    print(_get_forced_chunk_set("緯來體育台報導土地公有政策?？還是土地婆有政策。.", dicti))
    word_to_weight = {
        "土地公": 1,
        "土地婆": 1,
        "公有": 2,
        "": 1,
        "來亂的": "啦",
        "緯來體育台": 1,
    }
    dictionary = construct_dictionary(word_to_weight)
    print(dictionary)
    word_sentence = ["緯來", "體育台", "報導", "土地", "公有", "政策", "?？", "還是", "土地", "婆", "有", "政策", "。", "."]
    print(_run_word_segmentation_with_dictionary(word_sentence, recommend_dictionary=dictionary))
    print(_run_word_segmentation_with_dictionary(word_sentence, coerce_dictionary=dictionary))
