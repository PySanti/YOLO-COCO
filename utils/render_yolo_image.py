
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
