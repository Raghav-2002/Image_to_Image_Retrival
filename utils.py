import numpy as np
from glob import glob
import torch
import clip
import PIL
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_img_dir(dir_path):
  i = 0
  Encodings = []
  images_path = np.sort(glob(dir_path))
  for path in images_path:
    i = i + 1
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
      image_features = model.encode_image(image).detach().cpu().numpy()[0]
      Encodings.append(image_features)
  np.save(dir_path.split("/")[-2]+".npy",np.array(Encodings))

def encode_text(text_list):
  Encoding = []
  for text in text_list:
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
      text_features = model.encode_text(text).detach().cpu().numpy()[0]
      Encoding.append(text_features)
  np.save("Actions.npy",np.array(Encoding))



