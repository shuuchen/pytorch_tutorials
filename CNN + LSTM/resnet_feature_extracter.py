import torch
import torchvision.transforms as transforms
from PIL import Image


class Img2Vec():

    def __init__(self, model_path='./fine_tuning_dict.pt'):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model = torch.load(model_path) # because the model was trained on a cuda machine
        else:
            self.model = torch.load(model_path, map_location='cpu')

        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model = self.model.to(self.device)
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img_path, tensor=True):

        image = self.normalize(self.to_tensor(self.scaler(Image.open(img_path)))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]
