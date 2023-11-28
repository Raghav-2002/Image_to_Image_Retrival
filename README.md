## Creating Environment 

```.bash
conda create --name I2I
conda activate I2I
```
Install pytorch from : [pytorch.org](https://pytorch.org/)
```.bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn
```


## Preparing Datasets

Add `human_activity_retrieval_dataset.zip` to the `Data` folder
```.bash
mkdir Data
cp human_activity_retrieval_dataset.zip Data/
unzip human_activity_retrieval_dataset.zip
```

## Trained Model

Weights of the trained model i.e. MLP can be downloaded from [here](https://drive.google.com/file/d/12V8kzt5nv3AyoK_wZldQNeJ5lgqnhxmL/view?usp=sharing) 

## Encode Images and Text

1) `train.npy` : It has CLIp Space Image embeddings of train images . PLease download it from [here](https://drive.google.com/file/d/1wjKJS2bY03_sO0Xw9VOe3DplwJXaBaqM/view?usp=sharing)
2) `query_images.npy` : It has CLIP Space Image embeddings of query images
3) `gallery.npy`: It has CLIP Space Image embeddings of gallery images
4) `Actions.npy`: It has CLIP Space text embeddings of the 15 labels

NOTE : This step can be skipped if you wish to use the provided `.npy` files. Running the below code will simply overwrite the provided `.npy` files. 

To encode images in train, test and gallery folders run the following

```.bash
python encode_folder.py
```

## Training Model

```.bash
python Train.py
```

## Evaluvate Model

To Evaluvate Model run and display thr mean average precision 

```.bash
python Evaluvate.py 
```