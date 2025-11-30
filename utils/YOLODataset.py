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
