import os
import tensorflow as tf
import numpy as np

def load_texts_from_folder(folder, n=0):
    texts = []
    txt_files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    if n >= len(txt_files):
        print(f"Index {n} hors limite. Chargement du premier fichier.")
        n = 0
    selected_files = [txt_files[n]]
    print(f"Fichier chargé: {selected_files} avec le numéro: {n}")

    for filename in selected_files:
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

def tokenize_texts(texts, tokenizer):
    tokenized = []
    for text in texts:
        tokens = tokenizer.tokenize(text).numpy().tolist()
        tokenized.extend(tokens)
    return tokenized

def make_dataset(tokenized_LIST, vocab, maxlen, id=0, min_sequences=50000, num_UNKS_max=10, return_id=False):
    print("Début ID :", id)
    sequences = []
    
    try:
        UNK_index = vocab.index("[UNK]")
    except ValueError:
        UNK_index = 3 

    total_seen = 0
    total_accepted = 0

    while len(sequences) < min_sequences:
        if id >= len(tokenized_LIST):
            id = 0 

        tokenized = tokenized_LIST[id]
        i = 0
        while i < len(tokenized) - maxlen:
            seq = tokenized[i:i + maxlen + 1]
            unk_count = np.sum(np.array(seq) == UNK_index)
            total_seen += 1
            
            # Rejet si trop de UNK ou si les derniers tokens sont des UNK
            if unk_count <= num_UNKS_max and all(tok != UNK_index for tok in seq[-10:]):
                sequences.append(seq)
                total_accepted += 1

            if len(sequences) >= min_sequences:
                break
            i += 1

        if i >= len(tokenized) - maxlen:
            id += 1
            accepted_percent = 100 * total_accepted / total_seen if total_seen > 0 else 0
            print(f"Séquences acceptées (ID précédent) : {total_accepted} / {total_seen} ({accepted_percent:.2f}%)")
            total_seen = 0
            total_accepted = 0

    sequences = np.array(sequences)
    np.random.shuffle(sequences)

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    accepted_percent = 100 * total_accepted / total_seen if total_seen > 0 else 0
    print(f"Total acceptées : {total_accepted} / {total_seen} ({accepted_percent:.2f}%)")

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    if return_id:
        return dataset, id
    return dataset