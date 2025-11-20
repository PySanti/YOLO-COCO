from ultralytics import YOLO

# Cargamos un modelo pre-entrenado (solo para inicializar)
model = YOLO('yolov8n.pt') 

# Al intentar entrenar con 'coco.yaml', el sistema descargará todo automáticamente
# data='coco.yaml' le dice a YOLO que busque el dataset estándar COCO
# epochs=1 es solo para que el proceso corra y descargue; luego puedes cancelar
model.train(data='coco.yaml', epochs=1, imgsz=640)
