import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import register_keras_serializable

@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        initializer = tf.keras.initializers.RandomUniform(seed=np.random.randint(10000))
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, kernel_initializer=initializer)

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu", kernel_initializer=initializer),
            layers.Dropout(0.1),
            layers.Dense(embed_dim, kernel_initializer=initializer),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=False, attention_mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_config = config.copy()
        kwargs = {k: base_config[k] for k in list(base_config.keys()) if k not in {
            "vocab_size", "maxlen", "embed_dim", "num_heads", "ff_dim", "num_layers"
        }}
        return cls(
            embed_dim=base_config["embed_dim"],
            num_heads=base_config["num_heads"],
            ff_dim=base_config["ff_dim"],
            **kwargs
        )

@register_keras_serializable()
class GPTModel(keras.Model):
    def __init__(self, vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        initializer = tf.keras.initializers.RandomUniform(seed=np.random.randint(10000))
        self.token_emb = layers.Embedding(vocab_size, embed_dim, embeddings_initializer=initializer)
        self.pos_emb = layers.Embedding(maxlen, embed_dim, embeddings_initializer=initializer)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dropout = layers.Dropout(0.1)
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.out = layers.Dense(vocab_size, kernel_initializer=initializer)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        positions = tf.expand_dims(positions, 0)
        x = self.token_emb(x) + positions
        x = self.dropout(x, training=training)

        # Masque causal
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len)) 

        for block in self.blocks:
            x = block(x, training=training, attention_mask=mask)
        x = self.ln(x)
        return self.out(x)

    def build(self, input_shape):
        super().build(input_shape)
        self.token_emb.build(input_shape)
        self.pos_emb.build((input_shape[0], self.maxlen))