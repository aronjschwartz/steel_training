# steel_training
Project for a steel defect classification Kaggle Competition



Last edit: 8/19/2019 by Aron Schwartz

1.) Need the following in same directory:

mask_rcnn_coco.h5      	   #Coco pre-trained weights 
datasets/train/            #Folder of training images
datasets/val/              #Folder of validation images
steel_labels.txt           #Stores the class names (types 1,2,3,4)
steel_train.py             #Primary training file
train.csv                  #CSV containing core data


Change ROOT DIR on line 19 to your directory in steel_train.py

Run the following from command line to execute the program (while in working directory as shown above):

python ./steel_train.py train --dataset=./datasets --weights=coco