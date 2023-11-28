from Train import MLP
import torch
import numpy as np
from numpy.linalg import norm
import json
from sklearn.metrics import classification_report
from glob import glob


# Initializing MLP
mlp = MLP()
mlp.load_state_dict(torch.load("MLP.pt"))
mlp.eval()

# Function to return cosine similarity of IMage embeddings after passing through MLP
def get_similarity(vec_1,vec_2):

  vec_1 = vec_1[None,:]
  vec_2 = vec_2[None,:]
  prob1 = mlp(torch.tensor(vec_1)).detach().cpu().numpy()[0]
  prob2 = mlp(torch.tensor(vec_2)).detach().cpu().numpy()[0]
  cs = np.dot(prob1,prob2)/(norm(prob1)*norm(prob2))
  return cs

def test(K):
  
  # List of Gallery Images
  Gallery_list = np.sort(glob("Data/human_activity_retrieval_dataset/gallery/*"))
  # # List of Query Images
  Query_list = np.sort(glob("Data/human_activity_retrieval_dataset/query_images/*"))
  gallery_imgs = np.load("gallery.npy")
  query_imgs = np.load("query_images.npy")
  test_dict = json.load(open("Data/human_activity_retrieval_dataset/test_image_info.json","r"))
  Actions = set( val for val in test_dict.values())
  i = 0
  y_pred = []
  y_true = []
  Act = []
  for ac in Actions:
    Act.append(ac)
  Act = np.sort(np.array(Act))
  Actions = np.sort(Act)
  flag = 0
  print("Evaluvating Queries for K=",K)
  for qur_index in range(0,len(query_imgs)):
    
    qur = query_imgs[qur_index]
    i = i +1
    Similarity = []
    for gal_index in range(0,len(gallery_imgs)):
      gal = gallery_imgs[gal_index]
      siml = get_similarity(gal,qur)
      Similarity.append(siml)


    max_index = np.argmax(Similarity)
    y_true.append(list(Actions).index(test_dict[Query_list[qur_index].split("/")[-1]]))


    K_max_index = np.argpartition(Similarity, -K)[-K:]


    for pred_index in K_max_index:
        qur = Query_list[qur_index].split("/")[-1]
        gal = Gallery_list[pred_index].split("/")[-1]
        if test_dict[qur] == test_dict[gal]:
          flag = 1
          break
    if flag:
      y_pred.append(list(Actions).index(test_dict[Query_list[qur_index].split("/")[-1]]))
    else:
      y_pred.append(list(Actions).index(test_dict[Gallery_list[max_index].split("/")[-1]]))
    flag = 0
  target_names = Actions
  print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == "__main__":
  print("Metrics for K=1")
  test(K=1)
  print("Metrics for K=10")
  test(K=10)
  print("Metrics for K=50")
  test(K=50)
