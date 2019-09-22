---------------------- READ ME ---------------------------------
## authors: 
    Sarah Curtit, Caroline Pascal 

## usage: 
    python TP1.py coord_1 coord_2 coord_3 coord_a coord_b coord_c
the coord arguments are optional and indicate which features to display when displaying the datasets
coord_1 to coord_3 are the coordinates of the haberman's dataset's features - Indexed from 0 to 2
coord_a to coord_c are the coordinates of the breast-cancer's dataset's features - Indexed from 0 to 8
If the coordinates are not specified or are out of range, the first three features for each dataset will be displayed

## output
    For each dataset, this program will choose the best number of neighbors (<=20) for performing a knn classification algorithm, using accuracy as a criteria
For the best value of k, it will display the confusion matrix, the dataset's predictions and the accuracy evolution


