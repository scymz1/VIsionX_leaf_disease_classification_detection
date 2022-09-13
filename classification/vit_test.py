import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import TrivialAugmentWide
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
from transformers import MobileViTFeatureExtractor, MobileViTModel
from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests
import argparse

parser = argparse.ArgumentParser(description='input file path you want to evaluate')
parser.add_argument('--num_class', type=int, default=38,
                    help='number of class to classify')

parser.add_argument('--img', type=str,
                    help='number of class to classify')

parser.add_argument('--pretrained_model', type=str, default="./Mobilevit_leaf_classify.pth",
                    help='number of class to classify')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

idx_to_class = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MobileViTForImageClassification2(nn.Module):

    def __init__(self, num_labels):

        super(MobileViTForImageClassification2, self).__init__()
        #self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = MobileViTModel.from_pretrained("apple/mobilevit-small")
        #print(self.vit.config)
        self.classifier = nn.Linear(640*7*7, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):

        outputs = self.vit(pixel_values=pixel_values)
        #print(outputs.last_hidden_state.shape)
        flatterned = torch.flatten(outputs.last_hidden_state, start_dim=1)
        logits = self.classifier(flatterned)
        #print(logits)
        #last_hidden_states=outputs.last_hidden_state
        #logits = self.classifier(last_hidden_states[:,0,:])
        return logits

model=MobileViTForImageClassification2(num_labels=args.num_class)
model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device(device)))
model.to(device)
model.eval()


img = Image.open(args.img)

input = transform(img)
input = input.unsqueeze(0).to(device)
outputs = model(input)
_, predictions = torch.max(outputs, 1)
index = predictions.item()
print("predicted result :" + idx_to_class[index])