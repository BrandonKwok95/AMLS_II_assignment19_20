# Dogs & Cat Classifier based on LeNet-5 & VCG16

Here, it's an instruction on how to use files in the folder, so It's recommended to download comlete project on Google Drive
https://drive.google.com/open?id=1mxxJQk5HSuGzTwPlOeOQjptfiPlJi8Xf.

It's assignment in AMLS_II
The project include Three parts: LeNet-5, Vcg16, Datasets, main.py.

## 1. Datasets
Since the datasets is too large, it is difficult to upload whole dataset on Github. However, you can download the dataset on Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data. In order to use the project, you should put the dowloaded folder 'train' & 'test1' into 'Datasets'
![Image text](https://github.com/BrandonKwok95/AMLS_II_assignment19_20/blob/master/project_display.png)
![Image text](https://github.com/BrandonKwok95/AMLS_II_assignment19_20/blob/master/Datasets_display.png)

## 2. LeNet-5 & VCG16
After prepared Datasets, you can run the 'train_model.py' in A_LeNet-5 and B_VCG16 to train model and you can get result model in folder 'model'. Moreover, you camn run the 'test_model.py' to receieve a '.csv' file, which can be uploaded on Kaggle to get a test score.(https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/submissions)
Here is the score on two models:
![Image text](https://github.com/BrandonKwok95/AMLS_II_assignment19_20/blob/master/socre.png)

## 3. main.py
Since the train time consume too much time, so here I just display the result based on trained model created in above steps.
