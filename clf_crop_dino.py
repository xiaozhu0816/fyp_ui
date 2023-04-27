from torchvision import models, transforms
import torch
from torch import nn
from PIL import Image
import numpy as np
import vision_transformer as vits
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def predict(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
    url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    # state_dict = torch.load('./dino_vitbase16_pretrain.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    embed_dim = 3072
    
    transform = transforms.Compose([
                                    # transforms.Resize(256, interpolation=3),
                                    transforms.Resize([224,224], interpolation=3),
                                    # transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
    )])

    img = image
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    intermediate_output = model.get_intermediate_layers(batch_t, 4)
    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

    out_list = []
    weights_root = './weights'
    weights_types = ['chihen', 'coatcolor', 'coatni', 'coatthickness', 'dianci', 'liewen']
    text_root = './text'
    for weights_type in weights_types:
        weights_path = weights_root + '/' + 'best_expand_' + weights_type + '_dino.pth'
        linear_classifier = LinearClassifier(embed_dim, num_labels=2)
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        state_dict_new = linear_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict.items(), state_dict_new.items()):
            name = k_0
            state_dict_new[name] = v
        linear_classifier.load_state_dict(state_dict_new, strict=False)
        linear_classifier.eval()
        out = linear_classifier(output)
        with open(text_root + '/' + weights_type + '.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        out_list.append([(classes[idx], prob[idx].item()) for idx in indices[0][:1]])

    return out_list

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
