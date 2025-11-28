# YOLO-COCO


El objetivo de este proyecto es poner en practica la arquitectura de red neuronal YOLO para deteccion de objetos dentro de imagenes.

En este proyecto se implementara la arquitectura desde 0 utilizando el dataset COCO.



#   Informacion del dataset

Para descargar el dataset, se accedio a la ![pagina oficial de coco](https://cocodataset.org/#download) y se descargaron los siguientes archivos:

![Imagen 1](./images/image1.png)


```
2017 Train images [118K/18GB]
2017 Val images [5K/1GB]
2017 Train/Val annotations [241 MB]
```

Se creo la clase `YOLODataset` para wrappear el dataset.

```python

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.get_image_target import get_image_target

class YOLODataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X # paths list
        self.Y = Y # target wrapper

    def __getitem__(self, idx) :
        """
            Recordar que las imagenes tienen unos ids (que se encuentra en su nombre)
            Mientras que los targets tienen otro id
        """
        image_path = self.X[idx]
        image_id = int(image_path.split('/')[-1].split('.')[0])
        image = Image.open(self.X[idx]).convert('RGB')
        image_tensor = transforms.ToTensor()(image)
        image.close()

        return image_tensor, get_image_target(image_id, self.Y)
    def __len__(self):
        return len(self.X)
```

# Normalizacion y estandarizacion

Despues de investigar, revisamos que en YOLO no suelen estandarizarse las imagenes, sin embargo, si se suelen normalizar, trabajo que ya hace `transforms.ToTensor()`

# Revision de targets

#   Entrenamiento

#   Evaluacion

