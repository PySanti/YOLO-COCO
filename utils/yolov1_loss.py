import torch
from utils.yolo_iou import yolo_iou


def _to_abs_xywh_from_cell(box_tx_ty_tw_th: torch.Tensor, S: int) -> torch.Tensor:
    """
    Convierte cajas en formato YOLOv1 "por celda" a formato absoluto normalizado a la imagen.

    ENTRADA (formato por celda):
      box_tx_ty_tw_th = [..., 4] donde cada caja es [tx, ty, tw, th]

      - tx, ty: coordenadas del centro RELATIVAS a la celda (rango típico 0..1)
               Ej: tx=0.3 significa “30% del ancho de la celda desde el borde izquierdo de la celda”.
      - tw, th: ancho/alto NORMALIZADOS respecto a la imagen completa (0..1).

    SALIDA (formato absoluto normalizado a la imagen):
      [..., 4] donde cada caja es [x_abs, y_abs, w_abs, h_abs]
      - x_abs, y_abs: centro normalizado en coordenadas de la imagen completa (0..1).
      - w_abs, h_abs: igual a tw, th (ya estaban normalizados a imagen).

    Soporta dos shapes comunes:
      - target_box: (B, S, S, 4)
      - pred_box:   (B, S, S, A, 4)   (A = cantidad de anchors por celda)
    """
    device = box_tx_ty_tw_th.device
    B = box_tx_ty_tw_th.shape[0]  # batch size

    # ------------------------------------------------------------
    # Creamos un "grid" con los índices de celda (i, j) por cada posición.
    # i recorre columnas (eje x) -> 0..S-1
    # j recorre filas    (eje y) -> 0..S-1
    #
    # gx y gy quedan del shape (B, S, S) para que podamos hacer:
    #   x_abs = (i + tx)/S
    #   y_abs = (j + ty)/S
    # ------------------------------------------------------------
    gx = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)  # (B,S,S) columnas
    gy = torch.arange(S, device=device).view(1, S, 1).expand(B, S, S)  # (B,S,S) filas

    # ------------------------------------------------------------
    # Si las cajas tienen dimensión de anchors (B,S,S,A,4), necesitamos
    # “alinear” gx, gy para que puedan broadcast con esa dimensión A.
    # Pasan a (B,S,S,1) y se broadcast a (B,S,S,A) automáticamente.
    # ------------------------------------------------------------
    if box_tx_ty_tw_th.ndim == 5:
        gx = gx.unsqueeze(3)  # (B,S,S,1)
        gy = gy.unsqueeze(3)  # (B,S,S,1)

    # ------------------------------------------------------------
    # Separación de componentes:
    # tx,ty (centro relativo a la celda)
    # tw,th (tamaño normalizado a imagen)
    # ------------------------------------------------------------
    tx = box_tx_ty_tw_th[..., 0]
    ty = box_tx_ty_tw_th[..., 1]
    tw = box_tx_ty_tw_th[..., 2]
    th = box_tx_ty_tw_th[..., 3]

    # ------------------------------------------------------------
    # Conversión EXACTA pedida:
    # - sumamos el índice de celda (i o j) al offset dentro de la celda (tx, ty)
    # - dividimos entre S para normalizar a la imagen completa.
    #
    # Ejemplo (S=7):
    #   si estamos en la celda i=3 y tx=0.5,
    #   x_abs = (3 + 0.5)/7 = 0.5 -> centro a mitad de la imagen (aprox).
    # ------------------------------------------------------------
    x_abs = (gx + tx) / S
    y_abs = (gy + ty) / S
    w_abs = tw
    h_abs = th

    # Reconstruimos el tensor [...,4] con el mismo orden
    return torch.stack((x_abs, y_abs, w_abs, h_abs), dim=-1)


def yolov1_loss(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    """
    Loss estilo YOLOv1 (adaptado a tu head), donde:
      - Predices 2 "anchors" por celda.
      - Cada anchor predice: [x,y,w,h,conf, clases]
      - Las clases se predicen POR ANCHOR (por eso el last_dim es 2*(5+C) = 190 con C=90)

    Shapes esperados:
      predictions: (B, S, S, 2*(5+C))  -> 190 si C=90
        anchor1: [x,y,w,h, conf, class(0..C-1)]
        anchor2: [x,y,w,h, conf, class(0..C-1)]

      targets: (B, S, S, 5+C)
        [tx,ty,tw,th, tconf, onehot_classes...]

    Nota importante de consistencia:
      - El target guarda (tx,ty) relativo a celda, pero (tw,th) normalizado a imagen.
      - Para calcular IoU correctamente, convertimos a coordenadas ABS normalizadas
        con _to_abs_xywh_from_cell() antes de llamar a yolo_iou().
    """

    # ------------------------------------------------------------
    # Extraemos dimensiones:
    # B: batch
    # S: número de celdas por eje (grid SxS)
    # P: tamaño del vector por celda en predictions
    # ------------------------------------------------------------
    B, S, _, P = predictions.shape

    C = num_classes
    A = 2                       # anchors por celda
    stride = 5 + C              # tamaño por anchor: 4 box + 1 conf + C clases
    expected_P = A * stride     # 2*(5+C)

    # Validación defensiva: evita bugs silenciosos por mismatch en el head o en el dataset
    if P != expected_P:
        raise ValueError(
            f"Predictions last dim debe ser {expected_P} (= {A}*(5+{C})) pero llegó {P}."
        )
    if targets.shape[-1] != (5 + C):
        raise ValueError(
            f"Targets last dim debe ser {5+C} (= 5+{C}) pero llegó {targets.shape[-1]}."
        )

    # ------------------------------------------------------------
    # Partimos predictions en 2 bloques (uno por anchor):
    # p1: (B,S,S, 5+C)
    # p2: (B,S,S, 5+C)
    # ------------------------------------------------------------
    p1 = predictions[..., 0:stride]
    p2 = predictions[..., stride:2 * stride]

    # Anchor 1
    pred_box1 = p1[..., 0:4]          # (B,S,S,4) -> [tx,ty,tw,th] en tu espacio de predicción
    pred_conf1 = p1[..., 4]           # (B,S,S)
    pred_cls1 = p1[..., 5:5 + C]      # (B,S,S,C)

    # Anchor 2
    pred_box2 = p2[..., 0:4]
    pred_conf2 = p2[..., 4]
    pred_cls2 = p2[..., 5:5 + C]

    # ------------------------------------------------------------
    # Targets:
    # target_box: (B,S,S,4) -> [tx,ty,tw,th]
    # target_conf: (B,S,S) -> 1 si hay objeto asignado a esa celda, 0 si no
    # target_cls: (B,S,S,C) -> one-hot
    # ------------------------------------------------------------
    target_box = targets[..., 0:4]
    target_conf = targets[..., 4]
    target_cls = targets[..., 5:5 + C]

    # Máscaras booleanas:
    # obj_mask: celdas con objeto
    # noobj_mask: celdas sin objeto
    obj_mask = target_conf > 0
    noobj_mask = ~obj_mask

    # ------------------------------------------------------------
    # IoU consistente:
    # Convertimos pred y target a [x_abs,y_abs,w_abs,h_abs] en (0..1) respecto a la imagen.
    # Así el IoU no mezcla escalas de celda vs imagen (bug común).
    # ------------------------------------------------------------
    pred_abs1 = _to_abs_xywh_from_cell(pred_box1, S)     # (B,S,S,4)
    pred_abs2 = _to_abs_xywh_from_cell(pred_box2, S)     # (B,S,S,4)
    targ_abs = _to_abs_xywh_from_cell(target_box, S)     # (B,S,S,4)

    # IoU por celda (solo mide geometría, no conf ni clase)
    iou1 = yolo_iou(pred_abs1, targ_abs)  # (B,S,S)
    iou2 = yolo_iou(pred_abs2, targ_abs)  # (B,S,S)

    # Para cada celda, decidimos qué anchor explica mejor el target (mayor IoU)
    best_is_2 = iou2 > iou1  # (B,S,S) True => anchor2 es el responsable

    # ------------------------------------------------------------
    # Elegimos, por celda, cuál anchor es el responsable:
    # - resp_box / resp_conf / resp_cls => del anchor ganador
    # - nonresp_conf => del anchor perdedor (para penalizarlo en celdas con objeto)
    # ------------------------------------------------------------
    best_is_2_exp = best_is_2.unsqueeze(-1)  # (B,S,S,1) para poder hacer where en tensores [...,4] o [...,C]

    resp_box = torch.where(best_is_2_exp, pred_box2, pred_box1)   # (B,S,S,4)
    resp_conf = torch.where(best_is_2, pred_conf2, pred_conf1)    # (B,S,S)
    resp_cls = torch.where(best_is_2_exp, pred_cls2, pred_cls1)   # (B,S,S,C)

    nonresp_conf = torch.where(best_is_2, pred_conf1, pred_conf2) # (B,S,S)

    # ------------------------------------------------------------
    # COMPONENTES DE LA LOSS
    #
    # 1) Box regression:
    #    Se calcula SOLO donde hay objeto (obj_mask).
    #    Compara en el “espacio de entrenamiento” [tx,ty,tw,th],
    #    o sea, el mismo formato del target.
    # ------------------------------------------------------------
    box_loss = lambda_coord * torch.sum(((resp_box - target_box) ** 2)[obj_mask])

    # ------------------------------------------------------------
    # 2) Confidence:
    #    - obj_loss: el anchor responsable debe predecir conf ~ 1 (target_conf=1)
    #    - noobj_loss: penaliza conf alto en:
    #        a) celdas sin objeto (ambos anchors)
    #        b) el anchor NO responsable en celdas con objeto (debe “callarse”)
    # ------------------------------------------------------------
    obj_loss = torch.sum((resp_conf[obj_mask] - target_conf[obj_mask]) ** 2)

    noobj_loss = lambda_noobj * (
        torch.sum((pred_conf1[noobj_mask]) ** 2) +
        torch.sum((pred_conf2[noobj_mask]) ** 2) +
        torch.sum((nonresp_conf[obj_mask]) ** 2)
    )

    # ------------------------------------------------------------
    # 3) Classification:
    #    SOLO donde hay objeto.
    #    Como tu modelo predice clases por anchor (y no una sola vez por celda),
    #    usamos las clases del anchor responsable.
    # ------------------------------------------------------------
    class_loss = torch.sum(((resp_cls - target_cls) ** 2)[obj_mask])

    # Suma total
    return box_loss + obj_loss + noobj_loss + class_loss
