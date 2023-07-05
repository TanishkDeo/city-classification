import pickle
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.hub import load_state_dict_from_url, tqdm
from torchvision import datasets, models
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with the standard resnet101
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                models.resnet.model_urls['resnet101'],
                progress=True
            )
            self.load_state_dict(state_dict)

    # Reimplementing forward pass.
    # Replacing the forward inference defined here
    # http://tiny.cc/23pmmz
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

root = "data"

# transform the data so they are identical shapes
transform = transforms.Compose([transforms.Resize((255, 255)),
                                 transforms.ToTensor()])
dataset = ImageFolderWithPaths(root, transform=transform)

# load the data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=80, shuffle=True)


# initialize model
model = ResNet101(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# initialize variables to store results
features = None
labels = []
image_paths = []

# run the model
for batch in tqdm(dataloader, desc='Running the model inference'):

  images = batch[0].to('cpu')
  labels += batch[1]
  image_paths += batch[2]

  output = model.forward(images)
  # convert from tensor to numpy array
  current_features = output.detach().numpy()

  if features is not None:
      features = np.concatenate((features, current_features))
  else:
      features = current_features

# return labels too their string interpretations
labels = [dataset.classes[e] for e in labels]

# save the data
np.save('images.npy', images)
np.save('features.npy', features)
with open('labels.pkl', 'wb') as f:
  pickle.dump(labels, f)
with open('image_paths.pkl', 'wb') as f:
  pickle.dump(image_paths, f)



with open("image_paths.pkl", 'rb') as f:
    image_paths = pickle.load(f)

with open("images.npy", 'rb') as f:
    images = np.load(f)

with open("features.npy", 'rb') as f:
    features = np.load(f)
seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# run tsne
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(features)

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# scale and move the coordinates so they fit [0; 1] range
tx = scale_to_01_range(tsne_result[:,0])
ty = scale_to_01_range(tsne_result[:,1])

# plot the images

def compute_plot_coordinates(image, x, y):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(500 * x) + 400

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(500 * (1 - y)) + 400

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tlx = center_x - int(image_width / 2)
    tly = center_y - int(image_height / 2)

    brx = tlx + image_width
    bry = tly + image_height

    return tlx, tly, brx, bry


tsne_plot = 255 * np.ones((1000, 1000, 3), np.uint8)


c = dict()

for image_path, image, x, y in zip(image_paths, images, tx, ty):
  # read the image
  image = cv2.imread(image_path)

  # resize the image
  image = cv2.resize(image, (100,100))

  # compute the dimensions of the image based on its tsne co-ordinates
  tlx, tly, brx, bry = compute_plot_coordinates(image, x, y)

  n = f"{image_path}"
  n = n[n.rfind("/") + 1 : n.rfind(".")]
  c[n] = (round(x * 100), round(100 * y))

  # put the image to its t-SNE coordinates using numpy sub-array indices
  tsne_plot[tlx:brx, tly:bry, :] = image



cv2.imshow('t-SNE', tsne_plot)
cv2.waitKey()






h = [u[0] for u in c.values()]
v = [u[1] for u in c.values()]


fig, ax = plt.subplots()
ax.scatter(h, v)

for i, (k,j) in enumerate(c.items()):
    ax.annotate(k, (h[i], v[i]))

plt.show()



# initialize a list to capture a parallel set of labels
# so instead of the country, we can label our data through writing system, etc.
