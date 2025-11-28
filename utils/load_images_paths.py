from os import listdir
from os import path


def load_images_paths(folder : str):
    return [path.join(folder,i) for i in listdir(folder)]
