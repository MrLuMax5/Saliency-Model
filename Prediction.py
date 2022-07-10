import os

import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize

from ProjectDataset import TestDataset
from ProjectDataLoader import FixationDataLoader
from ProjectModel import EncoderASPPDecoder

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
composed = Compose([ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "path-to-dir"
image_file = "path-to-file"

test_data = TestDataset(root_dir=root_dir,
                        image_file=image_file,
                        image_transform=composed)
test_data_loader = FixationDataLoader(data=test_data,
                                      batch_size=8)

model = EncoderASPPDecoder()
model.to(device)
model.load_state_dict(torch.load('results/model.pt')['model'])

# Generate predictions for test images. Use same procedure as in the evaluation step
predictions = []
for batch in test_data_loader:
    image = batch['image'].to(device)
    with torch.no_grad():
        predictions.append(torch.softmax(model(image), dim=0))
predictions = torch.vstack(predictions)  # shape: (1032, 1, 224, 224)

path = 'results/test_fixations/'
os.makedirs(path, exist_ok=True)
with open(root_dir, 'r') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        name = '%sprediction-%s' % (path, line.split('-')[1])
        image = torchvision.transforms.ToPILImage()(predictions[i])
        image.save(name)
