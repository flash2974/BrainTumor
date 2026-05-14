from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Reshape,
    Layer,
    Add,
)
from keras.models import Sequential, Model
import tensorflow as tf

__all__ = ["ViT"]


def patch_encoder(patch_size: int, embedding_dimension: int) -> Sequential:
    """
    Découpe l'image d'entrée en plein de petits carrés (patch) et les projette dans un espace vectoriel.

    Args:
        patch_size: Dimension des patchs carrés (ex: 16 pour des patchs 16x16).
        embedding_dimension: Dimension du vecteur de sortie (D).

    Returns:
        Sequential: Modèle faisant le pipeline.
    """
    # La Conv2D agit comme une projection linéaire appliquée localement
    conv_layer = Conv2D(
        filters=embedding_dimension,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )
    # Aplatit les dimensions spatiales (H, W) en une seule dimension de séquence (N)
    reshape_layer = Reshape((-1, embedding_dimension))

    return Sequential([conv_layer, reshape_layer], name="patch_encoder")


class PatchLayer(Layer):
    """
    Prépare la séquence pour le Transformer en ajoutant le CLS token et les positions.

    Cette couche effectue deux opérations cruciales :
    1. Concatène un jeton de classe (CLS) au début de la séquence.
    2. Ajoute un biais apprenable (Positional Encoding) pour conserver l'ordre spatial.
    """

    def __init__(self, embedding_dimension: int, dropout_rate):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.dropout = Dropout(dropout_rate)

    def build(self, input_shape: list[int]):
        # CLS Token : vecteur unique de taille (D) qui accumulera l'info globale
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embedding_dimension),
            initializer="zeros",  # Souvent initialisé à 0 ou petit aléatoire
            trainable=True,
        )

        # Positional Encoding : matrice (1, N+1, D) ajoutée pour briser l'invariance par permutation
        num_patches = input_shape[-2]
        self.p_enc = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches + 1, self.embedding_dimension),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Broadcasting du CLS token sur la taille du batch
        batch_size = tf.shape(x)[0]
        brd_cls = tf.tile(self.cls, [batch_size, 1, 1])

        # x passe de (B, N, D) à (B, N+1, D)
        x = tf.concat([brd_cls, x], axis=1)

        # Injection de l'information spatiale
        x = x + self.p_enc

        return self.dropout(x, training=training)


class TransformerBlock(Layer):
    """
    Bloc d'encodeur Transformer standard (Pre-Normalization).

    Applique successivement la Self-Attention multi-têtes et un réseau
    Feed-Forward (MLP) avec des connexions résiduelles.
    """

    def __init__(self, nb_heads: int, embedding_dimension: int, dropout_rate: float):
        super().__init__()
        self.dropout_mha = Dropout(dropout_rate)
        self.dropout_mlp = Dropout(dropout_rate)
        self.norm_mha = LayerNormalization(epsilon=1e-6)
        self.norm_mlp = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads=nb_heads, key_dim=embedding_dimension)
        self.add = Add()

        # MLP : Expansion vers 4*D puis réduction vers D
        self.dense1 = Dense(4 * embedding_dimension, activation="gelu")
        self.dense2 = Dense(embedding_dimension)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Branche Attention : x_norm -> MHA -> Residual
        x_norm = self.norm_mha(x)
        mha_out = self.mha(x_norm, x_norm, x_norm)
        mha_out = self.dropout_mha(mha_out, training=training)
        x = self.add([x, mha_out])

        # Branche MLP : x_norm -> Dense -> Dense -> Residual
        x_norm = self.norm_mlp(x)
        mlp_out = self.dense1(x_norm)
        mlp_out = self.dropout_mlp(mlp_out, training=training)
        mlp_out = self.dense2(mlp_out)
        return self.add([x, mlp_out])


class ViT(Model):
    def __init__(
        self,
        nb_classes: int,
        embedding_dimension: int = 128,
        patch_size: int = 16,
        nb_heads: int = 8,
        nb_blocs: int = 6,
        dropout_rate: float = 0.1,
    ):
        """
        Vision Transformer

        Architecture :
        Input -> PatchEncoder -> PatchLayer (CLS + Pos) -> N * TransformerBlocks -> Head
        """
        super().__init__()
        self.name = "ViT"

        self.data_augmentation = Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.1),  # Tourne de +/- 10% max
                RandomZoom(0.1),
            ]
        )

        self.encoder = patch_encoder(patch_size, embedding_dimension)
        self.patch_layer = PatchLayer(embedding_dimension, dropout_rate)

        # Empilement des blocs d'encodeurs
        self.blocks = Sequential(
            [
                TransformerBlock(nb_heads, embedding_dimension, dropout_rate)
                for _ in range(nb_blocs)
            ],
            name="transformer_encoder",
        )

        self.norm = LayerNormalization(epsilon=1e-6)
        self.class_head = Dense(
            nb_classes, activation="softmax", name="classification_head"
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if training:
            x = self.data_augmentation(x)

        # Transformation de l'image en séquence positionnée
        x = self.encoder(x)
        x = self.patch_layer(x, training=training)

        # Passage dans le tunnel de blocs Transformer
        x = self.blocks(x, training=training)

        # on récupère le token CLS
        cls_token = x[:, 0, :]
        cls_token = self.norm(cls_token)

        return self.class_head(cls_token)
