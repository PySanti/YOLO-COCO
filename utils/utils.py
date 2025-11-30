import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import listdir
from os import path
import matplotlib.pyplot as plt
import numpy as np
from utils.MACROS import ANNOTATIONS_REQUIRED
import os



def get_dataset_classes_count(paths, target_wrapper):
    """
        Dado una lista de paths de imagenes
        muestra informacion acerca de la distribucion
        de sus targets
    """
    # esta version de coco contiene 90 clases
    train_class_dist = [0 for i in range(91)]
    for path in paths:
        image_ann = get_image_target(get_image_id(path), target_wrapper)
        for bbox in image_ann:
            cat = int(bbox["category_id"])
            train_class_dist[cat] +=1
    return train_class_dist


def get_image_id(image_path):
    filename = os.path.basename(image_path)          # '000000391895.jpg'
    stem, _ = os.path.splitext(filename)             # '000000391895'
    return int(stem)

def get_image_target(image_id, target_wrapper):
    """
        Retorna el target de la imagen a partir de su ID
    """
    ann_ids = target_wrapper.getAnnIds(imgIds=[image_id]) # se obtiene el id de la anotacion a partir de la imagen
    annotations = target_wrapper.loadAnns(ann_ids) # se obtienen las anotaciones
    return [{x:y for x,y in a.items() if x in ANNOTATIONS_REQUIRED} for a in annotations]

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

def plot_class_distribution(class_counts, class_names=None, title="Distribución de Clases"):
    """
    class_counts : list o np.array
        Conteo de apariciones por clase (indexado por class_id).
    
    class_names : list o None
        Nombres de las clases en el mismo orden que class_counts.
        Si es None, se usan los índices como etiquetas.
    
    title : str
        Título del gráfico.
    """

    class_counts = np.array(class_counts)

    # Filtrar solo clases con apariciones > 0
    valid_idx = np.where(class_counts > 0)[0]
    valid_counts = class_counts[valid_idx]

    if class_names is not None:
        valid_labels = [class_names[i] for i in valid_idx]
    else:
        valid_labels = [str(i) for i in valid_idx]

    plt.figure(figsize=(14, 6))
    plt.bar(valid_labels, valid_counts)
    plt.title(title)
    plt.xlabel("Clase")
    plt.ylabel("Número de instancias")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



def load_images_paths(folder : str):
    return [path.join(folder,i) for i in listdir(folder)]
