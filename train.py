import os
import csv
import argparse
import random
import numpy as np

# === CONFIGURATION DES LOGS TENSORFLOW ===
debug = False  #

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




from model import GPTModel
from data import load_texts_from_folder, tokenize_texts, make_dataset

# Configuration GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def log_training_stats(file_path, file_data, epoch, LR, loss, val_loss):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['file', 'epoch', 'LR', 'loss', 'val_loss'])
        writer.writerow([file_data, epoch, LR, loss, val_loss])

def get_last_log_entry(file_path):
    if not os.path.isfile(file_path):
        return None, None, None, None, None
    with open(file_path, mode='r') as file:
        lines = list(csv.reader(file))
        if len(lines) <= 1:
            return None, None, None, None, None
        last_line = lines[-1]
        return last_line[0], int(last_line[1]), float(last_line[2]), float(last_line[3]), float(last_line[4])

def main(args):
    # Création des dossiers
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vocab_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Récupération des logs pour reprise
    log_file = os.path.join(args.log_path, "training_log.csv")
    file_data, epoch, LR, loss, val_loss = get_last_log_entry(log_file)
    
    if file_data is None:
        file_data = 0
        epoch = 0
        LR = args.learning_rate
    
    debut_epoch = epoch + 1
    numero_fichiers = int(file_data)
    
    print(f"Reprise de l'entraînement : Fichier {numero_fichiers}, Epoch {debut_epoch}, LR {LR}")

    # Chargement du vocabulaire
    vocab_file = os.path.join(args.vocab_path, "vocab.txt")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    if len(vocab) > 3:
        vocab[3] = "[UNK]"
    vocab_size = len(vocab)
    print(f"Taille du vocabulaire : {vocab_size}")

    tokenizer = WordPieceTokenizer(vocabulary=vocab, lowercase=True)

    # Chargement des données initiales
    texts = load_texts_from_folder(args.data_path, n=numero_fichiers)
    tokenized = tokenize_texts(texts, tokenizer)
    
    # Découpage pour gestion de la mémoire
    taille_DATASET_Avec_UNK = int(args.taille_datasets / 0.35)
    tokenized_LIST = [
        tokenized[i * taille_DATASET_Avec_UNK : (i+1) * taille_DATASET_Avec_UNK] 
        for i in range((len(tokenized) // taille_DATASET_Avec_UNK) - 1)
    ]

    # Initialisation ou chargement du modèle
    checkpoint_filepath = os.path.join(args.checkpoint_dir, "model_checkpoint.keras")
    
    if args.creer_model or not os.path.exists(checkpoint_filepath):
        print("Création d'un nouveau modèle...")
        model = GPTModel(vocab_size, args.maxlen, args.embed_dim, args.num_heads, args.ff_dim, args.num_layers)
        model.build(input_shape=(None, args.maxlen))
    else:
        print(f"Chargement du modèle depuis {checkpoint_filepath}")
        model = keras.models.load_model(checkpoint_filepath, compile=False)

    optimizer = Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_freq='epoch'
    )

    NB = debut_epoch
    steps_per_epochs = args.taille_datasets // args.batch_size

    # Boucle d'entraînement continue
    print("=== DÉBUT DE L'ENTRAÎNEMENT ===")
    while True:
        print(f"\nCréation du dataset : chunk {NB} sur {len(tokenized_LIST)}")
        dataset, NB = make_dataset(tokenized_LIST, vocab, args.maxlen, id=NB, min_sequences=args.taille_datasets, return_id=True)
        
        train_dataset = dataset.shuffle(buffer_size=1024).batch(args.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        val_dataset = make_dataset(tokenized_LIST, vocab, args.maxlen, id=NB+1, min_sequences=args.taille_datasets)

        history = model.fit(
            train_dataset, 
            epochs=args.epochs, 
            steps_per_epoch=steps_per_epochs,
            validation_data=val_dataset.batch(args.batch_size),
            validation_steps=steps_per_epochs // 2,
            callbacks=[checkpoint_callback]
        )

        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]
        log_training_stats(log_file, numero_fichiers, NB, LR, last_loss, last_val_loss)

        NB += 1
        if NB >= len(tokenized_LIST) - 1:
            NB = 0
            numero_fichiers = (numero_fichiers + 1) % 42  
            
            # Recharger le texte suivant
            texts = load_texts_from_folder(args.data_path, n=numero_fichiers)
            tokenized = tokenize_texts(texts, tokenizer)
            tokenized_LIST = [
                tokenized[i * taille_DATASET_Avec_UNK : (i+1) * taille_DATASET_Avec_UNK] 
                for i in range((len(tokenized) // taille_DATASET_Avec_UNK) - 1)
            ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle MM1")
    # Dossiers
    parser.add_argument("--data_path", type=str, default="data_txt", help="Dossier des textes")
    parser.add_argument("--checkpoint_dir", type=str, default="CHECKPOINTS_MM1", help="Dossier des sauvegardes")
    parser.add_argument("--vocab_path", type=str, default="VOCAB", help="Dossier du vocabulaire")
    parser.add_argument("--log_path", type=str, default="log", help="Dossier des logs")
    
    # Modèle
    parser.add_argument("--creer_model", action="store_true", help="Forcer la création d'un nouveau modèle")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    
    # Entraînement
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--taille_datasets", type=int, default=50000)

    args = parser.parse_args()
    main(args)