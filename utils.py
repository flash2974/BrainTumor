from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CWD = Path('.')
YES_PATH = CWD / "data" / "yes"
NO_PATH = CWD / "data" / "no"
DATASETS_DIR = CWD / "datasets"
SIZE = 128

def load_image(path):
    img = Image.open(path).convert('L')
    # image non étirée (bandes nories)
    img_padded = ImageOps.pad(img, (SIZE, SIZE), method=Image.Resampling.LANCZOS, color=0)
    
    return np.array(img_padded)


def load_dataset() :
    yes_images = np.array([load_image(path) for path in YES_PATH.iterdir()])
    yes_labels = np.ones(yes_images.shape[0])
    
    no_images = np.array([load_image(path) for path in NO_PATH.iterdir()])
    no_labels = np.zeros(no_images.shape[0])
    
    ds_images = np.concatenate((yes_images, no_images))
    ds_labels = np.concatenate((yes_labels, no_labels))
    
    idx = np.random.permutation(len(ds_images))
    
    X = ds_images[idx].reshape((-1, SIZE*SIZE))
    y = ds_labels[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)
    X_train = X_train.reshape((-1, SIZE, SIZE, 1))
    X_test = X_test.reshape((-1, SIZE, SIZE, 1))
    
    return X_train, X_test, y_train, y_test

def plot_image(arr) : 
    plt.imshow(arr, cmap='gray')
    plt.show()
    
    
    
class Dataset :
    def __init__(self, name) -> None:
        self.base_path = DATASETS_DIR / name
        self.s = StandardScaler()
        self.train = self.load_train()
        self.test = self.load_test()
        
        
    def load_train(self) :
        all_images = []
        all_labels = []
        
        path = self.base_path / "train"
        self.mapping = {}
        for i, category in enumerate(path.iterdir()):
            images = [load_image(p) for p in category.iterdir()]
            labels = [i] * len(images)
            
            all_images.extend(images)
            all_labels.extend(labels)
            self.mapping[i] = category.stem

        X = np.array(all_images)
        y = np.array(all_labels)
        
        self.num_classes = i + 1
        idx = np.random.permutation(len(X))
    
        X = X[idx].reshape((-1, SIZE*SIZE))
        y = y[idx]
        
        X = self.s.fit_transform(X)
        X = X.reshape((-1, SIZE, SIZE, 1))
        
        return X, y
    
    
    def load_test(self) :
        all_images = []
        all_labels = []
        
        path = self.base_path / "test"
        for i, category in enumerate(path.iterdir()):
            images = [load_image(p) for p in category.iterdir()]
            labels = [i] * len(images)
            
            all_images.extend(images)
            all_labels.extend(labels)
        
        X = np.array(all_images).reshape((-1, SIZE*SIZE))
        y = np.array(all_labels)
    
        idx = np.random.permutation(len(X))
    
        X = X[idx].reshape((-1, SIZE*SIZE))
        y = y[idx]
        
        X = self.s.transform(X)
        X = X.reshape((-1, SIZE, SIZE, 1))
        
        return X, y

    
    def reverse_transform(self, img) :
        flat = img.reshape((1, -1))
        raw = self.s.inverse_transform(flat)
        return raw.reshape((SIZE, SIZE))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        