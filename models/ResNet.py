from keras.models import Model, Sequential
from keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Activation,
    Layer,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

__all__ = ["ResNet"]


class ResNetBlock(Layer):
    def __init__(self, filters, strides, apply_shortcut=False):
        super().__init__()
        padding = "same"
        # Activation
        self.relu = Activation("relu")

        # First Conv
        self.conv1 = Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding)
        self.bn1 = BatchNormalization()

        # Second Conv
        self.conv2 = Conv2D(
            filters=filters, kernel_size=3, strides=strides, padding=padding
        )
        self.bn2 = BatchNormalization()

        # Third Conv
        self.conv3 = Conv2D(
            filters=filters * 4, kernel_size=1, strides=1, padding=padding
        )
        self.bn3 = BatchNormalization()

        # Le raccourci (shortcut) si les dimensions changent.
        self.shortcut = None
        if apply_shortcut or strides != 1:
            self.shortcut = Sequential(
                [
                    Conv2D(
                        filters * 4, kernel_size=1, strides=strides, padding=padding
                    ),
                    BatchNormalization(),
                ]
            )

        self.add = Add()  # Additionneur GPU

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        # Gestion du raccourci
        res = inputs
        if self.shortcut is not None:
            res = self.shortcut(inputs)

        if res.shape != x.shape:
            raise Exception(
                f"Size mismatch between res : {res.shape} and x : {x.shape}"
            )

        # Addition et activation finale
        x = self.add([res, x])
        x = self.relu(x)
        return x


class ResNet(Model):
    def __init__(self, num_classes: int):
        """Modèle de ResNet-50
        \\
        Config :
        - Entrée : Convolution 7x7 (stride 2) + BatchNorm + ReLU + MaxPool 3x3 (stride 2).
        - Stage 1 : 2 blocs Bottleneck (64 filtres internes, 256 en sortie).
        - Stage 2 : 2 blocs Bottleneck (128 filtres internes, 512 en sortie) avec réduction spatiale (stride 2).
        - Sortie : Global Average Pooling (GAP) suivi d'une couche Dense Softmax.

        Args:
            num_classes (int): Nombre de classes à prédire.
        """
        super().__init__()
        self.name = "ResNet"
        padding = "same"

        # Couche qui permet d'augmenter artificiellement la taille du dataset
        self.data_augmentation = Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.1),  # Tourne de +/- 10% max
                RandomZoom(0.1),
            ]
        )

        self.initial_conv = Conv2D(
            filters=64, kernel_size=7, strides=2, padding=padding
        )  # dimension des images / 2 avec stride
        self.initial_bn = BatchNormalization()
        self.initial_activ = Activation("relu")
        self.mp1 = MaxPooling2D(
            pool_size=3, strides=2, padding=padding
        )  # dimension des images / 4 avec stride

        # 1er bloc, on doit adapter le shortcut la sortie de mp1 comporte 64 filtres, or le conv3 du ResNet block en sort 256.
        # Une erreur se produira si on ne transforme pas l'entrée initiale en 256 filtres
        self.rs11 = ResNetBlock(filters=64, strides=1, apply_shortcut=True)
        self.rs12 = ResNetBlock(filters=64, strides=1)

        self.rs21 = ResNetBlock(filters=128, strides=2)
        self.rs22 = ResNetBlock(filters=128, strides=1)

        self.gap = GlobalAveragePooling2D()
        self.out = Dense(num_classes, activation="softmax")

    # Training : passé automatiquement quand on fait model.fit()
    def call(self, x, training=False):

        if training:
            x = self.data_augmentation(x)

        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_activ(x)
        x = self.mp1(x)

        x = self.rs11(x)
        x = self.rs12(x)

        x = self.rs21(x)
        x = self.rs22(x)

        x = self.gap(x)
        x = self.out(x)

        return x


"""
Explications :

Si on n'a pas ce Sequential (le shortcut) :
- rs11 reçoit en entrée notre bloc de 56x56x64. Il le sépare en deux chemins.
- Le raccourci (shortcut) : Il garde le bloc intact (56x56x64).
- Le chemin principal (les 3 convolutions) : La première conv (1x1) garde 64 canaux. 
  La deuxième (3x3) garde 64 canaux. Mais la troisième (1x1) a filters * 4. 
  Elle multiplie donc les canaux par 4 ! La sortie de ce chemin fait donc 56x56x256.
- Le crash à la fin de rs11 : À la toute fin du bloc rs11, on a la couche Add(). 
  Elle essaie d'additionner le raccourci (64 canaux) avec le chemin principal (256 canaux). 
  Erreur Keras immédiate, les dimensions ne correspondent pas.

Le Sequential est donc là pour "protéger" l'addition à l'intérieur de rs11. 
Il prend le raccourci de 64 canaux, lui applique une convolution 1x1 pour le gonfler à 256 canaux, et 
on peut enfin additionner 256 canaux avec 256 canaux !


Et rs12 dans tout ça ?

Lui, il a la vie facile. Puisque rs11 s'est débrouillé pour faire sortir un bloc de 256 canaux, 
rs12 reçoit 256 canaux en entrée.
Son chemin principal va faire ses calculs et produire à nouveau 256 canaux.
À la fin de rs12, le Add() va additionner son entrée (256 canaux) avec son chemin principal (256 canaux). 
Pas besoin de Sequential adaptateur, ça matche naturellement !
"""
