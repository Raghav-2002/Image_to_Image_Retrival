from glob import glob
import json
import numpy as np
from numpy.linalg import norm
import torch
from torch.utils.data import Dataset
import torch
from torch import nn
from utils import encode_text


class ImageDataset(Dataset):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]
        return image, label
    
def sim(vec_1,vec_2,mlp):
  vec_1 = vec_1[None,:]
  vec_2 = vec_2[None,:]
  prob1 = mlp(torch.tensor(vec_1)).detach().cpu().numpy()[0]
  prob2 = mlp(torch.tensor(vec_2)).detach().cpu().numpy()[0]
  cs = np.dot(prob1,prob2)/(norm(prob1)*norm(prob2))
  return cs



class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.Sigmoid()

    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

def loss_function(pred,target):

  loss = 0
  for idx in range(0,pred.size()[0]):
    pred_ = pred[idx].float()
    target_ = target[idx].float()
    loss += 1 - torch.dot(pred_,target_)/(torch.norm(pred_)*torch.norm(target_))
  return loss/pred.size()[0]

def train(X,Y,epochs):
   
  dataset = ImageDataset(X=X,Y=Y)

  trainloader = torch.utils.data.DataLoader(dataset, batch_size=5000, shuffle=True, num_workers=1)

  # Initialize the MLP 
  mlp = MLP()

  # Define the loss function and optimizer
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

  # Run the training loop
  print("Training Started...")
  for epoch in range(0, epochs): # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    step = 0

    for i, data in enumerate(trainloader, 0):

      # Get inputs
      step = step + 1
      inputs, targets = data
      # Zero the gradients
      optimizer.zero_grad()

      # Perform forward pass
      outputs = mlp(inputs)

      # Compute loss
      loss = loss_function(outputs, targets)

      # Perform backward pass
      loss.backward()

      # Perform optimization
      optimizer.step()

      # Print statistics
      current_loss += loss.item()

    print('Loss:',current_loss/step)
    if epoch % 50 == 0:
      print("Saving checkpoints...")
      torch.save(mlp.state_dict(), "MLP_"+str(current_loss/step)+".pt")

    print("Training Completed")
      
  # Process is complete.
  print('Training process has finished.')

if __name__ == "__main__":
  
    # Loading Encodings of gallery, query images and text embeddings of 15 class labels. Encoding is done using CLIP
    print("Loading Encodings")
    Gallery_imgs_encs = np.load("gallery.npy")
    Query_imgs_encs = np.load("query_images.npy")
    Actions_encs = np.load("Actions.npy")

    # Loading Json files
    print("Loading Json")
    test_dict = json.load(open("Data/human_activity_retrieval_dataset/test_image_info.json","r"))
    train_dict = json.load(open("Data/human_activity_retrieval_dataset/train_image_info.json"))

    print("Encoding Text Labels")
    Actions = set( val for val in test_dict.values())
    Act = []
    for ac in Actions:
       Act.append(ac)
    Act = np.sort(np.array(Act))
    encode_text(Act)

    print("Creating Train Set")
    Y = []
    X_fld = np.sort(glob("Data/human_activity_retrieval_dataset/train/*"))
    for idx in range(0,len(X_fld)):
      name = X_fld[idx].split("/")[-1]
      action = train_dict[name]
      idx_action = list(Act).index(action)
      Y.append(Actions_encs[idx_action])

    Y_train = np.array(Y)
    X_train = np.load("train.npy")

    train(X=X_train,Y=Y_train,epochs=1000)


    



    

