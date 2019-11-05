import os
from config import image_folder
import numpy as np

if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    np.random.shuffle(files)
    print(len(files))

    train = []
    valid = []
    test = []
