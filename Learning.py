import os

import torch
from torch.utils.data import RandomSampler
from torchvision.transforms import ToTensor, Compose, Normalize

from ProjectDataset import FixationDataset
from ProjectDataLoader import FixationDataLoader
from ProjectModel import EncoderASPPDecoder

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
composed_data_transforms = Compose([ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:%s" % device)

root_dir = "path-to-dir"
image_file = "path-to-file"
fixation_file = "path-to-fixations"

training_data = FixationDataset(root_dir=root_dir,
                                image_file=image_file,
                                fixation_file=fixation_file,
                                image_transform=composed_data_transforms,
                                fixation_transform=ToTensor())
training_data_loader = FixationDataLoader(data=training_data,
                                          batch_size=16,
                                          sampler=RandomSampler(training_data))

validation_data = FixationDataset(root_dir=root_dir,
                                  image_file=image_file,
                                  fixation_file=fixation_file,
                                  image_transform=composed_data_transforms,
                                  fixation_transform=ToTensor())
validation_data_loader = FixationDataLoader(data=validation_data,
                                            batch_size=8)

model = EncoderASPPDecoder()
model.to(device)
os.makedirs('results/', exist_ok=True)
if(os.path.exists('results/model.pt')):
    print('Pre-trained model found. It is used for further training.')
    model.load_state_dict(torch.load('results/model.pt')['model'])

epochs = 32
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.KLDivLoss(reduction='batchmean')
for epoch in range(epochs):
    model.train()
    epoch_training_loss = 0
    for i, batch in enumerate(training_data_loader):
        image = batch['image'].to(device)
        target = batch['fixation'].to(device)
        prediction = torch.softmax(model(image), dim=0)
        loss = loss_function(prediction, target)
        epoch_training_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch %d, Batch %d: Loss %d" % (epoch, i, loss.item()))
    with open('results/losses.csv', 'a') as file:
        file.write("%s, %d\n" % ('training', epoch_training_loss // len(training_data_loader)))
    print("Training loss in epoch %d: %d" % (epoch, epoch_training_loss))

    model.eval()
    predictions = []
    ground_truths = []
    for batch in validation_data_loader:
        image = batch['image'].to(device)
        target = batch['fixation'].to(device)
        ground_truths.append(target)
        with torch.no_grad():
            predictions.append(model(image))
    predictions = torch.vstack(predictions)
    ground_truths = torch.vstack(ground_truths)
    # sizes:(len(dl), 1, 224, 224)
    epoch_validation_loss = loss_function(predictions, ground_truths).item()
    torch.save({'model': model.state_dict()}, f='results/model.pt')
    with open('results/losses.csv', 'a') as file:
        file.write("%s, %d\n" % ('validation', epoch_validation_loss // len(validation_data_loader)))
    print("Finish epoch %d with validation loss %d" % (epoch, epoch_validation_loss))
