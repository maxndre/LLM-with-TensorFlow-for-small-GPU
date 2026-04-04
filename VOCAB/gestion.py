import os
import re

input_vocab_path = "_MM1/VOCAB/vocab_cleaned.txt"
output_vocab_path = "_MM1/VOCAB/vocab_cleaned3.txt"

with open(input_vocab_path, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vocab_set = set(vocab)
cleaned_vocab = []
removed_words = []

for word in vocab:
    # Condition 1 : mot > 5 lettres, finit par "s", version sans "s" existe
    remove_plural = (
        len(word) > 5 and
        word.endswith("s") and
        not word.startswith("##") and
        word[:-1] in vocab_set
    )

    # Condition 2 : mot contient uniquement des chiffres
    remove_digits = bool(re.fullmatch(r"\d+", word))

    # Condition 3 : mot est du type ##1234 (sous-mot purement numérique)
    remove_digits_subword = bool(re.fullmatch(r"##\d+", word))

    if remove_plural or remove_digits or remove_digits_subword:
        removed_words.append(word)
    else:
        cleaned_vocab.append(word)

# Séparer mots d'un caractère et sous-mots '##' avec 1 caractère
single_char_or_subword = sorted([w for w in cleaned_vocab if (len(w) == 1) or (w.startswith("##") and len(w) == 3)])
other_words = [w for w in cleaned_vocab if not ((len(w) == 1) or (w.startswith("##") and len(w) == 3))]

# Sauvegarde
with open(output_vocab_path, "w", encoding="utf-8") as f:
    for word in single_char_or_subword + other_words:
        f.write(word + "\n")

# Affichage des mots supprimés
print("Mots supprimés :")
for word in removed_words:
    print(word)

print(f"\nTotal supprimés : {len(removed_words)}")
print(f"Vocabulaire original : {len(vocab)} tokens")
print(f"Vocabulaire nettoyé  : {len(single_char_or_subword) + len(other_words)} tokens")
print(f" - dont mots 1 caractère ou '##' + 1 caractère : {len(single_char_or_subword)}")
