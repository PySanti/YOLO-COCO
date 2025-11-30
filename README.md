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

Distribucion de ejemplos dispuestos:

```
Cantidad de elementos de train: 118287
Cantidad de elementos de val:   5000
```

# Normalizacion y estandarizacion

Despues de investigar, revisamos que en YOLO no suelen estandarizarse las imagenes, sin embargo, si se suelen normalizar, trabajo que ya hace `transforms.ToTensor()`

# Revision de targets

Despues de crear la clase `YOLODataset` ya obtuvimos acceso a generar tuplas `(image-tensor, target)` donde el target seria informacion relacionada con las bbox de las imagenes. Luego cree una funcion utilizada para, dada una imagen y dada sus annotations, renderizar la imagen + bboxes:


Teniendo en cuenta la siguiente lista de labels:

| ID | Nombre (Inglés) | Traducción (Español) |
|:--:|:----------------|:---------------------|
| 1  | person          | persona              |
| 2  | bicycle         | bicicleta            |
| 3  | car             | coche / auto         |
| 4  | motorcycle      | motocicleta          |
| 5  | airplane        | avión                |
| 6  | bus             | autobús              |
| 7  | train           | tren                 |
| 8  | truck           | camión               |
| 9  | boat            | bote                 |
| 10 | traffic light   | semáforo             |
| 11 | fire hydrant    | boca de incendios    |
| 13 | stop sign       | señal de stop        |
| 14 | parking meter   | parquímetro          |
| 15 | bench           | banco                |
| 16 | bird            | pájaro               |
| 17 | cat             | gato                 |
| 18 | dog             | perro                |
| 19 | horse           | caballo              |
| 20 | sheep           | oveja                |
| 21 | cow             | vaca                 |
| 22 | elephant        | elefante             |
| 23 | bear            | oso                  |
| 24 | zebra           | cebra                |
| 25 | giraffe         | jirafa               |
| 27 | backpack        | mochila              |
| 28 | umbrella        | paraguas             |
| 31 | handbag         | bolso                |
| 32 | tie             | corbata              |
| 33 | suitcase        | maleta               |
| 34 | frisbee         | frisbee              |
| 35 | skis            | esquís               |
| 36 | snowboard       | snowboard            |
| 37 | sports ball     | balón deportivo      |
| 38 | kite            | cometa               |
| 39 | baseball bat    | bate de béisbol      |
| 40 | baseball glove  | guante de béisbol    |
| 41 | skateboard      | monopatín            |
| 42 | surfboard       | tabla de surf        |
| 43 | tennis racket   | raqueta de tenis     |
| 44 | bottle          | botella              |
| 46 | wine glass      | copa de vino         |
| 47 | cup             | taza                 |
| 48 | fork            | tenedor              |
| 49 | knife           | cuchillo             |
| 50 | spoon           | cuchara              |
| 51 | bowl            | cuenco               |
| 52 | banana          | plátano / banana     |
| 53 | apple           | manzana              |
| 54 | sandwich        | sándwich             |
| 55 | orange          | naranja              |
| 56 | broccoli        | brócoli              |
| 57 | carrot          | zanahoria            |
| 58 | hot dog         | perrito caliente     |
| 59 | pizza           | pizza                |
| 60 | donut           | dona                 |
| 61 | cake            | pastel               |
| 62 | chair           | silla                |
| 63 | couch           | sofá                 |
| 64 | potted plant    | planta en maceta     |
| 65 | bed             | cama                 |
| 67 | dining table    | mesa de comedor      |
| 70 | toilet          | inodoro              |
| 72 | tv              | televisión           |
| 73 | laptop          | portátil             |
| 74 | mouse           | ratón                |
| 75 | remote          | control remoto       |
| 76 | keyboard        | teclado              |
| 77 | cell phone      | teléfono móvil       |
| 78 | microwave       | microondas           |
| 79 | oven            | horno                |
| 80 | toaster         | tostadora            |
| 81 | sink            | fregadero            |
| 82 | refrigerator    | refrigerador         |
| 84 | book            | libro                |
| 85 | clock           | reloj                |
| 86 | vase            | jarrón               |
| 87 | scissors        | tijeras              |
| 88 | teddy bear      | oso de peluche       |
| 89 | hair drier      | secador de pelo      |
| 90 | toothbrush      | cepillo de dientes   |



```python

# utils/render_yolo_image.py


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

def render_yolo_image(image_tensor, target):
    """
    image_tensor: Tensor [3, H, W] en rango [0,1]
    target: lista de dicts en formato COCO (annotations)
    """

    # Pasar tensor a formato HWC para matplotlib
    img = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for ann in target:
        # Formato COCO: [x_min, y_min, width, height]
        x, y, w, h = ann["bbox"]

        # Crear rectángulo
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        # Etiqueta de clase
        class_id = ann["category_id"]
        ax.text(
            x, y - 5,
            f"id:{class_id}",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.6)
        )

    ax.axis("off")
    plt.show()

```

Logrando este tipo de resultados:

![Imagen 2](./images/image2.png)
![Imagen 3](./images/image3.png)


# Entendiendo el formato de los targets

Las annotations tienen el siguiente formato:

```
{
  'segmentation': [...],   
  'area': 53481.5,        
  'iscrowd': 0,            
  'image_id': 42,          
  'bbox': [x, y, w, h],    
  'category_id': 18,      
  'id': 1817255
}
```

De los cuales, solo nos interesan los siguientes campos:

```
{
  'image_id': 42,           # ✅ útil para indexar
  'bbox': [x, y, w, h],     # ✅ FUNDAMENTAL
  'category_id': 18,        # ✅ FUNDAMENTAL
}
```

Es imporante tener en cuenta que los bbox contienen las coordenadas de las imagenes con valores absolutos a la imagen, y que llevan el siguiente formato:

```
bbox : [x_inicio, y_inicio, ancho, alto]
```

Por ejemplo, la ultima imagen mostrada, contiene el siguiente target:

```
[
{ image_id': 32129, 'bbox': [171.81, 196.61, 107.38, 140.67], 'category_id': 1}, 
{'image_id': 32129, 'bbox': [195.29, 332.36, 162.93, 31.61], 'category_id': 42}, 
{'image_id': 32129, 'bbox': [0.0, 72.5, 57.72, 96.8], 'category_id': 3}
]
```

Nota : se modifico el `get_image_target` para solo retornar los campos que requerimos ...

```python

from utils.MACROS import ANNOTATIONS_REQUIRED

def get_image_target(image_id, target_wrapper):
    """
        Retorna el target de la imagen a partir de su ID
    """
    ann_ids = target_wrapper.getAnnIds(imgIds=[image_id]) # se obtiene el id de la anotacion a partir de la imagen
    annotations = target_wrapper.loadAnns(ann_ids) # se obtienen las anotaciones
    return [{x:y for x,y in a.items() if x in ANNOTATIONS_REQUIRED} for a in annotations]


```



#   Entrenamiento

#   Evaluacion

