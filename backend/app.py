import flask
import io
import requests
from PIL import Image
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections
import torchvision.transforms as transforms

app = flask.Flask(__name__)
CORS(app)
class Net(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        for param in self.model1.parameters():
            param.requires_grad = False

        for param in self.model1.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 2)
        )

    def forward(self, x, y, z):
        x1 = self.model1(x)
        x2 = self.model2(y)
        x3 = self.model3(z)

        combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)), dim=1)
        return combined

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

modelset = [
    ('efficientNet-b7', [torchvision.models.efficientnet_b7(pretrained=True), torchvision.models.efficientnet_b7(pretrained=True), torchvision.models.efficientnet_b7(pretrained=True)]),
]

models = modelset[0][1]
model = Net(models[0], models[1], models[2])
for name, param in model.named_parameters():
    param.requires_grad = False

my_classifier = nn.Sequential(
    nn.Linear(3000, 500),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(500, 2)
)
model = mySequential(collections.OrderedDict([
    ('net', model),
    ('classifier', my_classifier)
]))

print('Loading...')
# TODO: pth 파일 저장 후 불러오기
model.load_state_dict(torch.load('./SGI_demo.pth', map_location='cpu'))
print('Loaded!')
model.eval()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    image.save('hello.png')
    img = cv2.imread('hello.png')
    img = cv2.resize(img, dsize=(800, 800))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = np.transpose(img, (2, 0, 1))
    # print(img.shape)
    img = img.astype(np.float32) / 255.0

    transform = transforms.ToTensor()
    img = transform(img)

    # print(img)
    return img.unsqueeze(0)

def get_prediction(mastoid, glabella, supraorbital):
    outputs = model.forward(mastoid, glabella, supraorbital)

    return (torch.argmax(outputs, dim=1).detach().numpy(), outputs.detach().numpy())

@app.route('/')
def status():
    return 'OK'

@app.route('/estimation', methods=["POST"])
def estimation():
    if flask.request.method == "POST":
        params = flask.request.get_json()

        if (params['mastoid'] != ''):
            res = requests.get(params['mastoid'])
            mastoid = transform_image(res.content)

        if (params['glabella'] != ''):
            res = requests.get(params['glabella'])
            glabella = transform_image(res.content)


        if (params['supraorbital'] != ''):
            res = requests.get(params['supraorbital'])
            supraorbital = transform_image(res.content)

        res = get_prediction(mastoid, glabella, supraorbital)

        return flask.jsonify({
            'gender': res[0][0],
            'score_0': res[1][0][0],
            'score_1': res[1][0][1]
        })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)
