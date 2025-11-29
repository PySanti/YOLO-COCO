from utils.MACROS import ANNOTATIONS_REQUIRED

def get_image_target(image_id, target_wrapper):
    """
        Retorna el target de la imagen a partir de su ID
    """
    ann_ids = target_wrapper.getAnnIds(imgIds=[image_id]) # se obtiene el id de la anotacion a partir de la imagen
    annotations = target_wrapper.loadAnns(ann_ids) # se obtienen las anotaciones
    return [{x:y for x,y in a.items() if x in ANNOTATIONS_REQUIRED} for a in annotations]


