from torch.utils.data import Dataset
from utils.utils import get_image_id
from torchvision import transforms
from PIL import Image
from utils.utils import get_image_target

class YOLODataset(Dataset):
    def __init__(self, X, Y, transformer) -> None:
        super().__init__()
        self.X = X # paths list
        self.Y = Y # target wrapper
        self.transformer = transformer

    def __getitem__(self, idx) :
        """
            Recordar que las imagenes tienen unos ids (que se encuentra en su nombre)
            Mientras que los targets tienen otro id
        """
        image = Image.open(self.X[idx]).convert('RGB') 
        image_tensor = self.transformer(image)
        image.close()
        image_annotation = get_image_target(get_image_id(self.X[idx]), self.Y)
        return image_tensor, image_annotation

    def __len__(self):
        return len(self.X)
