from utils import encode_img_dir, encode_text
import numpy as np
import json

encode_img_dir("Data/human_activity_retrieval_dataset/gallery/*")
encode_img_dir("Data/human_activity_retrieval_dataset/query_images/*")
encode_img_dir("Data/human_activity_retrieval_dataset/train/*")

test_dict = json.load(open("Data/human_activity_retrieval_dataset/test_image_info.json","r"))
train_dict = json.load(open("Data/human_activity_retrieval_dataset/train_image_info.json"))

Actions = set( val for val in test_dict.values())
Act = []
for ac in Actions:
    Act.append(ac)
Act = np.sort(np.array(Act))
encode_text(Act)
