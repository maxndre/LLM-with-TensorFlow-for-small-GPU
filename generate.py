import os
import csv
import argparse
import random
import numpy as np

# === CONFIGURATION DES LOGS TENSORFLOW ===
debug = False  

if not debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    import logging
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_nlp.tokenizers import WordPieceTokenizer





from model import TransformerBlock, GPTModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_text(prompt, model, tokenizer, vocab, maxlen=128, max_length=127, temperature=1.0):
    tokens = tokenizer.tokenize(tf.constant([prompt]))[0].numpy().tolist()  
    if len(tokens) > maxlen - 1:
        tokens = tokens[-(maxlen-1):]

    # Padding
    input_tokens = tokens + [0] * (maxlen - len(tokens))
    input_tokens = np.array([input_tokens])
    generated = tokens.copy()

    try:
        UNK_index = vocab.index("[UNK]")
    except ValueError:
        UNK_index = 3

    print(f"--- Génération en cours (Température : {temperature}) ---")

    for _ in range(max_length):
        preds = model.predict(input_tokens, verbose=0)
        
        index_to_use = min(len(generated) - 1, maxlen - 1)
        next_token_logits = preds[0, index_to_use]

        # Bannir le token UNK
        next_token_logits[UNK_index] = -1e15

        next_token_probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        next_token = np.random.choice(len(next_token_probs), p=next_token_probs)

        generated.append(next_token)

        if len(generated) > maxlen:
            input_tokens = np.array([generated[-maxlen:]])
        else:
            input_tokens = np.array([generated + [0] * (maxlen - len(generated))])

    full_text = tokenizer.detokenize(tf.convert_to_tensor([generated]))[0].numpy().decode("utf-8")
    return full_text

def main(args):
    # Charger vocabulaire
    vocab_file = os.path.join(args.vocab_path, "vocab.txt")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    
    tokenizer = WordPieceTokenizer(vocabulary=vocab, lowercase=True)

    # Charger Modèle
    checkpoint_filepath = os.path.join(args.checkpoint_dir, "model_checkpoint.keras")
    print(f"Chargement du modèle depuis {checkpoint_filepath}...")
    
    model = keras.models.load_model(
        checkpoint_filepath, 
        custom_objects={"TransformerBlock": TransformerBlock, "GPTModel": GPTModel},
        compile=False
    )

    # Génération
    resultat = generate_text(args.prompt, model, tokenizer, vocab, maxlen=args.maxlen, max_length=args.length, temperature=args.temperature)
    
    print("\n" + "="*50)
    print("PROMPT :", args.prompt)
    print("-" * 50)
    print("RÉSULTAT :\n")
    print(resultat)
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de texte MM1")
    parser.add_argument("--prompt", type=str, required=True, help="Le début de la phrase à compléter")
    parser.add_argument("--checkpoint_dir", type=str, default="CHECKPOINTS_MM1", help="Dossier du modèle")
    parser.add_argument("--vocab_path", type=str, default="VOCAB", help="Dossier du vocabulaire")
    parser.add_argument("--maxlen", type=int, default=128, help="Taille du contexte du modèle")
    parser.add_argument("--length", type=int, default=50, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Créativité (ex: 0.7 pour plus strict, 1.2 pour fou)")
    
    args = parser.parse_args()
    main(args)