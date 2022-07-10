from torch.utils.data import Dataset
import skimage.io
import os


def read_text_file(filename: str):
	lines = []
	with open(filename, 'r') as file:
		for line in file: 
			line = line.strip()  # Or some other preprocessing
			lines.append(line)
	return lines


class FixationDataset(Dataset):
	def __init__(self, root_dir: str, image_file: str, fixation_file,
				 image_transform=None, fixation_transform=None):
		self.root_dir = root_dir
		self.image_files = read_text_file(image_file)
		self.fixation_files = None if fixation_file is None else read_text_file(fixation_file)
		self.image_transform = image_transform
		self.fixation_transform = fixation_transform

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.image_files[idx])
		image = skimage.io.imread(img_name)

		fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
		fix = skimage.io.imread(fix_name)

		sample = {"image": image, "fixation": fix}

		if self.image_transform:
			sample["image"] = self.image_transform(sample["image"])
		if self.fixation_transform:
			sample["fixation"] = self.fixation_transform(sample["fixation"])

		return sample


class TestDataset(Dataset):
	def __init__(self, root_dir: str, image_file: str, image_transform=None):
		self.root_dir = root_dir
		self.image_files = read_text_file(image_file)
		self.image_transform = image_transform

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.image_files[idx])
		image = skimage.io.imread(img_name)
		sample = {"image": image}
		if self.image_transform:
			sample["image"] = self.image_transform(sample["image"])
		return sample