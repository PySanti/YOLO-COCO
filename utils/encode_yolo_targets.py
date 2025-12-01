
import torch

def encode_yolo_target(
    annotations,
    image_width,
    image_height,
    grid_size,
    num_classes,
):
    """
    Convierte anotaciones COCO de UNA imagen a un tensor target tipo YOLO.

    Parameters
    ----------
    annotations : list[dict]
        Lista de anotaciones de una imagen en formato COCO.
        Cada dict debe tener, al menos:
            - "bbox": [x, y, w, h] en píxeles (formato COCO, esquina sup. izq.)
            - "category_id": id de clase (COCO)
    image_width : int
        Ancho de la imagen (en píxeles).
    image_height : int
        Alto de la imagen (en píxeles).
    grid_size : int
        Tamaño S de la grilla (S x S).
    num_classes : int
        Número de clases del dataset.

    Returns
    -------
    target : torch.Tensor
        Tensor [S, S, 5 + num_classes] con:
            target[..., 0:4] = [tx, ty, tw, th]
            target[..., 4]   = confidence (0 o 1)
            target[..., 5:]  = one-hot de clases
    """

    S = grid_size
    C = num_classes

    # [S, S, 5 + C]
    target = torch.zeros((S, S, 5 + C), dtype=torch.float32)

    cell_w = image_width / S
    cell_h = image_height / S

    for ann in annotations:
        bbox = ann["bbox"]     # [x, y, w, h] en píxeles
        x, y, w, h = bbox

        # Centro del bbox en píxeles
        x_c = x + w / 2.0
        y_c = y + h / 2.0

        # Índice de la celda donde cae el centro
        i = int(x_c / cell_w)  # columna (eje x)
        j = int(y_c / cell_h)  # fila (eje y)

        # Ignorar cajas fuera de la imagen o justo en el borde extremo
        if i < 0 or i >= S or j < 0 or j >= S:
            continue

        # Coordenadas relativas a la celda (entre 0 y 1)
        x_cell = x_c / cell_w
        y_cell = y_c / cell_h
        tx = x_cell - i
        ty = y_cell - j

        # Ancho y alto normalizados al tamaño completo de la imagen
        tw = w / image_width
        th = h / image_height

        # Si la celda ya tiene un objeto, podemos decidir si reemplazarlo
        # por el más grande (en área) para no perder información.
        if target[j, i, 4] == 1:
            # Ya hay objeto → comparamos áreas
            prev_tw = target[j, i, 2] * image_width
            prev_th = target[j, i, 3] * image_height
            prev_area = prev_tw * prev_th
            new_area = w * h

            # Si el que ya está es más grande, nos lo quedamos
            if prev_area >= new_area:
                continue

        # Guardamos bbox normalizado y confianza
        target[j, i, 0] = tx
        target[j, i, 1] = ty
        target[j, i, 2] = tw
        target[j, i, 3] = th
        target[j, i, 4] = 1.0

        # One-hot de clase (asumiendo COCO: category_id ~ [1..num_classes])
        cat_id = ann["category_id"]
        class_idx = cat_id - 1  # si tus clases van de 1 a C

        if 0 <= class_idx < C:
            target[j, i, 5 + class_idx] = 1.0

    return target
