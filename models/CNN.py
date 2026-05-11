from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
import tensorflow as tf

__all__ = ["CNN"]

class CNN(Model) : 
    def __init__(self, num_classes : int) :
        """CNN basique.
        \\
        Couches :
            - Conv2D(32, 4) -> MaxPooling()
            - Conv2D(64, 4) -> MaxPooling()
            - GlobalAveragePooling()
            - Dense(128)
            - Dropout(.3)
            - Dense finale (classification head)

        Args:
            num_classes (int): Nombre de classes à prédire.
        """
        super().__init__()  
        self.name = "CNN"
        self.seq = Sequential([
            Conv2D(32, 4, activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 4, activation='relu'),
            MaxPooling2D(),
            
            GlobalAveragePooling2D(),
            Dense(128, activation = 'relu'),
            Dropout(0.3),
            Dense(num_classes, activation = 'softmax')
        ])
        
    # Appelée lors d'un model.predict()
    def call(self, x : tf.Tensor) -> tf.Tensor: 
        return self.seq(x)