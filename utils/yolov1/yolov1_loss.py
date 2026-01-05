import torch
from utils.MACROS import BATCH_SIZE, GRID_SIZE, NUM_CLASSES
from utils.yolo_iou import yolo_iou
import torch.nn.functional as F


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

def allocate_prediction_yolov1(predictions, targets, num_classes=NUM_CLASSES):
    """
        Toma un tensor generado por un modelo YOLOv1 y lo retorna luego de implementar el
        algoritmod de asignacion
    """
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

    p1 = predictions[..., 0:stride]
    p2 = predictions[..., stride:2 * stride]

    pred_box1 = p1[..., 0:4]          # (B,S,S,4) -> [tx,ty,tw,th] en tu espacio de predicción
    pred_conf1 = p1[..., 4]           # (B,S,S)
    pred_cls1 = p1[..., 5:5 + C]      # (B,S,S,C)

    # Anchor 2
    pred_box2 = p2[..., 0:4]
    pred_conf2 = p2[..., 4]
    pred_cls2 = p2[..., 5:5 + C]

    # separacion de componentes para el target
    target_box = targets[..., 0:4]

    pred_abs1 = _to_abs_xywh_from_cell(pred_box1, S)     # (B,S,S,4)
    pred_abs2 = _to_abs_xywh_from_cell(pred_box2, S)     # (B,S,S,4)
    targ_abs = _to_abs_xywh_from_cell(target_box, S)     # (B,S,S,4)

    # IoU por celda (solo mide geometría, no conf ni clase)
    iou1 = yolo_iou(pred_abs1, targ_abs)  # (B,S,S)
    iou2 = yolo_iou(pred_abs2, targ_abs)  # (B,S,S)

    best_is_2 = iou2 > iou1  # (B,S,S) True => anchor2 es el responsable

    best_is_2_exp = best_is_2.unsqueeze(-1)  # (B,S,S,1) para poder hacer where en tensores [...,4] o [...,C]

    resp_box = torch.where(best_is_2_exp, pred_box2, pred_box1)   # (b,s,s,4)
    resp_conf = torch.where(best_is_2, pred_conf2, pred_conf1)    # (b,s,s)
    resp_cls = torch.where(best_is_2_exp, pred_cls2, pred_cls1)   # (b,s,s,c)
    return torch.cat((resp_box, resp_conf.unsqueeze(-1), resp_cls), dim=-1), (iou1, iou2)

def yolov1_loss(
    preds: torch.Tensor,     # (B, S, S, 5*B + C)
    targets: torch.Tensor,   # (B, S, S, 5 + C)  -> [tx,ty,w,h,obj] + onehot(C)
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.1,
    eps: float = 1e-9,
    anchors=2
):
    """
    Loss YOLOv1 estable:
    - coord: MSE en tx,ty y en sqrt(w), sqrt(h) SOLO para la bbox responsable.
    - obj: BCEWithLogits(conf_logit, iou_detached) SOLO bbox responsable en celdas con obj.
    - noobj: BCEWithLogits(conf_logit, 0) para:
        * todas las bboxes en celdas sin obj
        * bboxes NO responsables en celdas con obj
    - cls: CrossEntropy sobre logits de clase por celda (solo celdas con obj).
    """
    S= GRID_SIZE
    B= anchors
    C= NUM_CLASSES
    assert preds.dim() == 4 and targets.dim() == 4
    BS = preds.shape[0]
    assert preds.shape[1] == S and preds.shape[2] == S, "S no coincide con preds/targets"
    assert preds.shape[-1] == B*(5  + C), "dimensión final de preds incorrecta"
    assert targets.shape[-1] == 5 + C, "dimensión final de targets incorrecta"

    device = preds.device
    dtype = preds.dtype

    # ---- split preds
    box_raw = preds[..., : 5 * B].view(BS, S, S, B, 5)         # (BS,S,S,B,5)
    cls_logits = preds[..., 5 * B : 5 * B + C]                 # (BS,S,S,C)

    # raw -> constrained
    pred_tx_ty = torch.sigmoid(box_raw[..., 0:2])              # (BS,S,S,B,2) in [0,1]
    pred_wh = F.softplus(box_raw[..., 2:4])                    # (BS,S,S,B,2) > 0
    pred_conf_logit = box_raw[..., 4]                          # (BS,S,S,B) logits

    # ---- targets
    gt_tx_ty = targets[..., 0:2].to(dtype)                     # (BS,S,S,2)
    gt_wh = targets[..., 2:4].to(dtype)                        # (BS,S,S,2)
    obj_mask = (targets[..., 4] > 0.5)                         # (BS,S,S) bool

    # class target index (solo donde hay obj)
    gt_onehot = targets[..., 5:].to(dtype)                     # (BS,S,S,C)
    gt_cls_idx = gt_onehot.argmax(dim=-1)                      # (BS,S,S)

    # ---- helpers: convert (tx,ty,w,h) to absolute xyxy normalized [0,1]
    gy = torch.arange(S, device=device, dtype=dtype).view(1, S, 1, 1)  # row
    gx = torch.arange(S, device=device, dtype=dtype).view(1, 1, S, 1)  # col

    def to_xyxy_abs(tx_ty: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
        # tx_ty: (BS,S,S,B,2) ; wh: (BS,S,S,B,2)
        cx = (gx + tx_ty[..., 0]) / S
        cy = (gy + tx_ty[..., 1]) / S
        w = wh[..., 0].clamp(min=eps, max=1.0)
        h = wh[..., 1].clamp(min=eps, max=1.0)
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)  # (BS,S,S,B,4)

    def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (...,4) xyxy
        ax1, ay1, ax2, ay2 = a.unbind(dim=-1)
        bx1, by1, bx2, by2 = b.unbind(dim=-1)
        ix1 = torch.maximum(ax1, bx1)
        iy1 = torch.maximum(ay1, by1)
        ix2 = torch.minimum(ax2, bx2)
        iy2 = torch.minimum(ay2, by2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
        area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
        return inter / (area_a + area_b - inter + eps)

    # ---- compute IoU per predicted box vs GT box (in each cell)
    pred_xyxy = to_xyxy_abs(pred_tx_ty, pred_wh)               # (BS,S,S,B,4)

    # GT needs to be expanded to B boxes: (BS,S,S,1,2/4)
    gt_tx_ty_exp = gt_tx_ty.unsqueeze(-2).expand(-1, -1, -1, B, -1)
    gt_wh_exp = gt_wh.unsqueeze(-2).expand(-1, -1, -1, B, -1)
    gt_xyxy = to_xyxy_abs(gt_tx_ty_exp, gt_wh_exp)             # (BS,S,S,B,4)

    ious = iou_xyxy(pred_xyxy, gt_xyxy)                        # (BS,S,S,B)

    # responsible box = argmax IoU per cell
    best_box = ious.argmax(dim=-1)                             # (BS,S,S) long
    resp_mask = F.one_hot(best_box, num_classes=B).to(dtype=torch.bool)  # (BS,S,S,B)

    # ---- losses
    # Coord loss (solo responsible boxes en obj cells)
    obj_mask_b = obj_mask.unsqueeze(-1).expand(-1, -1, -1, B)   # (BS,S,S,B)
    resp_obj = resp_mask & obj_mask_b                           # (BS,S,S,B)

    # tx,ty MSE
    gt_tx_ty_b = gt_tx_ty.unsqueeze(-2).expand(-1, -1, -1, B, -1)
    coord_xy = F.mse_loss(pred_tx_ty[resp_obj], gt_tx_ty_b[resp_obj], reduction="sum") if resp_obj.any() else preds.new_tensor(0.0)

    # sqrt(w),sqrt(h) MSE (YOLOv1 classic)
    gt_wh_b = gt_wh.unsqueeze(-2).expand(-1, -1, -1, B, -1).clamp(min=eps, max=1.0)
    pred_wh_cl = pred_wh.clamp(min=eps, max=1.0)
    coord_wh = F.mse_loss(torch.sqrt(pred_wh_cl[resp_obj]), torch.sqrt(gt_wh_b[resp_obj]), reduction="sum") if resp_obj.any() else preds.new_tensor(0.0)

    loss_coord = lambda_coord * (coord_xy + coord_wh)

    # Objectness (responsible in obj cells): target = IoU (detached)
    if resp_obj.any():
        obj_target = ious.detach()[resp_obj]                    # (N,)
        loss_obj = F.binary_cross_entropy_with_logits(pred_conf_logit[resp_obj], obj_target, reduction="sum")
    else:
        loss_obj = preds.new_tensor(0.0)

    # No-objectness:
    # - all boxes in noobj cells
    # - plus boxes in obj cells that are NOT responsible
    noobj_cells = (~obj_mask).unsqueeze(-1).expand(-1, -1, -1, B)       # (BS,S,S,B)
    not_resp_obj = obj_mask_b & (~resp_mask)                             # (BS,S,S,B)
    noobj_mask_full = noobj_cells | not_resp_obj

    if noobj_mask_full.any():
        loss_noobj = F.binary_cross_entropy_with_logits(
            pred_conf_logit[noobj_mask_full],
            torch.zeros_like(pred_conf_logit[noobj_mask_full]),
            reduction="sum",
        )
    else:
        loss_noobj = preds.new_tensor(0.0)

    loss_noobj = lambda_noobj * loss_noobj

    # Class loss (solo celdas con obj): CrossEntropy sobre logits por celda
    if obj_mask.any():
        cls_logits_obj = cls_logits[obj_mask]                   # (N_obj, C)
        cls_target_obj = gt_cls_idx[obj_mask]                   # (N_obj,)
        loss_cls = F.cross_entropy(cls_logits_obj, cls_target_obj, reduction="sum")
    else:
        loss_cls = preds.new_tensor(0.0)

    total = (loss_coord + loss_obj + loss_noobj + loss_cls)

    stats = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_coord": float((loss_coord / max(BS, 1)).detach().cpu().item()),
        "loss_obj": float((loss_obj / max(BS, 1)).detach().cpu().item()),
        "loss_noobj": float((loss_noobj / max(BS, 1)).detach().cpu().item()),
        "loss_cls": float((loss_cls / max(BS, 1)).detach().cpu().item()),
        "obj_cells": int(obj_mask.sum().detach().cpu().item()),
    }
    return total
