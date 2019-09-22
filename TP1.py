import sys
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


#david.octavian.iacob@gmail.com

def compute_distances(data_train, data_test):
    """@brief compute euclidean distances between a single data point and a set of data points
    
    @param data_train : np array (n,m), the set of data points
    @param data_test : np array (m), the single data point
    
    @return np array (n) : 1-D array of all euclidean distances
    """
    diff = data_train[:,:-1]-data_test[:-1]
    return np.sqrt(np.sum(np.square(diff), axis=1))


def kNN_classification(data_train, data_test, k, dataset):
    """@brief performs a k-NN classification algorithm
    
    @param data_train : np array (n,m), set of labeled data
    @param data_test : np array (m), data to be classified
    @param k: int, number of neighbors to look for
    @param dataset: string, datasets we're currently working on

    @return int: if working with the haberman data set, 1 if the patient survived more than 5 years, 2 otherwise
                 if working with the breast tumor dataset, 2 if the tumor is begning, 4 if it is malign
    """

    distances = compute_distances(data_train, data_test)

    k_neighbors = np.argsort(distances)[1:k+1]
    neighbors_classes = data_train[k_neighbors, -1]

    if dataset == "haberman":
        if(len(np.where(neighbors_classes == 1)[0]) > k/2):
            return 1
        else:
            return 2
    else:
        if(len(np.where(neighbors_classes == 2)[0]) > k/2):
            return 2
        else:
            return 4


def plot_confusion_matrix(confusion_matrix, classes):
    """@brief plot the confusion matrix for the k-NN classification algorithm's results
    
    @param confusion_matrix: matrix to display
    @param classes: list of str, possible classes : ['1','2'] for haberman data set and ['2','4'] for breast tumor data set

    @return void, only plotting
    """
    confusion_matrix = np.around(confusion_matrix,decimals=3)

    #Choice of the Color Map
    cmap=plt.cm.RdPu

    #Plot
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    #Setting the axes with labels
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Normalized confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = confusion_matrix.max() / 4.
    thresh = 0.5
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def set_label(axis, coord, dataset, ax):
    """@brief set a given 3D plot's axis' label
    
    @param axis: axis to label, 0 if x-axis 1 if y-axis and 2 if z-axis
    @param coord: coordinate to display 
    @param dataset: string, name of the dataset we're currently working on
    @ax plt.axes object
    
    @return void 
    """
    if dataset == "haberman":
        if coord == 0:
            label = "Age of patient at time of operation"
        elif coord == 1:
            label = "Patient's year of operation - 1900"
        elif coord == 2:
            label = "Number of positive axillary nodes detected"
        else:
            plt.close()
            raise ValueError("Coordinate doesn't exist in the haberman dataset")
    else:
        if coord == 0:
            label = "Clump Thickness "
        elif coord == 1:
            label = "Uniformity of Cell Size"
        elif coord == 2:
            label = "Uniformity of Cell Shape"
        elif coord == 3:
            label = "Marginal Adhesion"
        elif coord == 4:
            label = "Single Epithelial Cell Size"
        elif coord == 5:
            label = "Bare Nuclei"
        elif coord == 6:
            label = "Bland Chromatin"
        elif coord == 7:
            label = "Normal Nucleoli"
        elif coord == 8:
            label = "Mitoses"
        else:
            plt.close()
            raise ValueError("Coordinate doesn't exist in the breast cancer dataset") 

    if axis == 0:
        ax.set_xlabel(label)
    elif axis == 1:
        ax.set_ylabel(label)
    elif axis == 2:
        ax.set_zlabel(label)



def try_k(data, k, dataset, nb_splits, plot=False, coord1=0, coord2=1, coord3=2):
    """performs a k-NN classification algorithm, computes accuracy and confusion matrices for a given number of neighbors
    
    @param data: numpy array (n,m) set of datas
    @param k: int, number of neighbors to look for
    @param dataset: string, name of the dataset we're currently working on
    @param plot: bool, whether the dataset should be displayed or not
    @param coord0: coordinate to display on x axis, default value is 0
    @param coord1: coordinate to display on y axis, default value is 1
    @param coord2: coordinate to display on z axis, default value is 2

    @return tupple, [accuracy error, confusion matrix]
    """
    
    #splitting the dataset in 3 parts. Data will be trained on two third and tested on the third third
    k_fold = sklearn.model_selection.KFold(n_splits=nb_splits, shuffle=True, random_state=None)

    total_confusion_matrix = np.zeros((2,2))
    total_accuracy = 0

    plot_counter=0

    cmap = plt.cm.get_cmap('Spectral')

    #seeking neighbors on 2/3 of the dataset, testing on 1/3
    for train_index, test_index in k_fold.split(data):
        if(plot):
            plot_counter+=1
            plt.figure(figsize=(12, 8))
            ax = plt.axes(projection="3d")

            set_label(0, coord1, dataset, ax)
            set_label(1, coord2, dataset, ax)
            set_label(2, coord3, dataset, ax)

            if(dataset=='haberman'):
                ax.set_title("Representation of the Haberman's survival data set - Test set number "+str(plot_counter))
                ax.plot3D([np.NaN],np.NaN,color=cmap(1/5+0.5),marker='o',label="class 1, prediction is correct",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(1/5+0.5),marker='*',label="class 1, prediction is false",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(2/5+0.5),marker='o',label="class 2, prediction is correct",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(2/5+0.5),marker='*',label="class 2, prediction is false",linestyle = 'None')

            else:
                ax.set_title("Representation of the breast cancer Wisconsin data set - Test set number "+str(plot_counter))
                ax.plot3D([np.NaN],np.NaN,color=cmap(2/5+0.5),marker='o',label="class 2, prediction is correct",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(2/5+0.5),marker='*',label="class 2, prediction is false",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(4/5+0.5),marker='o',label="class 4, prediction is correct",linestyle = 'None')
                ax.plot3D([np.NaN],np.NaN,color=cmap(4/5+0.5),marker='*',label="class 4, prediction is false",linestyle = 'None')

        confusion_matrix = np.zeros((2,2))
        
        #count indexes will be used to normalize the confusion matrix
        count_label1=0
        count_label2=0            

        #predicting labels on the test set, computing the confusion matrix
        for i in test_index:
            test_data = data[i]
            label = test_data[-1]

            if(dataset == 'haberman'):
                prediction = kNN_classification(data[train_index], test_data, k, dataset)
                if(label == 1):
                    count_label1+=1
                    if(prediction ==1):
                        confusion_matrix[0,0]+=1
                    else:
                        confusion_matrix[0,1]+=1
                else:
                    count_label2+=1
                    if(prediction ==1):
                        confusion_matrix[1,0]+=1
                    else:
                        confusion_matrix[1,1]+=1

            else:
                prediction = kNN_classification(data[train_index], test_data, k, dataset)
                if(label == 2):
                    count_label1+=1
                    if(prediction ==2):
                        confusion_matrix[0,0]+=1
                    else:
                        confusion_matrix[0,1]+=1
                else:
                    count_label2+=1
                    if(prediction ==2):
                        confusion_matrix[1,0]+=1
                    else:
                        confusion_matrix[1,1]+=1

            if(plot):
                ax.plot3D([data[i,0]],data[i,1],data[i,2], color=cmap(prediction/5+0.5), marker='o' if prediction==data[i,-1] else '*')
                plt.legend(loc='center left')


        confusion_matrix[0,:]/=count_label1
        confusion_matrix[1,:]/=count_label2
            
        total_accuracy+=(confusion_matrix[0,0]+confusion_matrix[1,1])/np.sum(confusion_matrix)
        total_confusion_matrix+=confusion_matrix

    total_confusion_matrix/=nb_splits
    total_accuracy/=nb_splits

    if(plot):
        plt.show()

    return [total_accuracy, total_confusion_matrix]
        

def search_best_k(max_k, dataset, coord1=0, coord2=1, coord3=2):
    """select best number of neighbors for performing a k-NN classification algorithm
    
    @param max_k: maximum number of neighbors for which to test
    @param dataset: string, datasets we're currently working on
    @param coord0: coordinate to display on x axis, default is 0
    @param coord1: coordinate to display on y axis, default is 1
    @param coord2: coordinate to display on z axis, default is 2

    @return void, display the confusion matrix, the dataset and accuracy evolution depending on k
    """


    #downloading dataset
    if(dataset == 'haberman'):
        data_haberman = np.loadtxt("haberman.data", comments="#", delimiter=",", unpack=False)
        data = data_haberman
    else:
        data_breast_cancer = np.loadtxt("breast-cancer-wisconsin.data", comments="#", delimiter=",", unpack=False, dtype='str')
        data_breast_cancer = np.delete(data_breast_cancer, np.where(data_breast_cancer == '?')[0], axis=0)
        data_breast_cancer =  data_breast_cancer.astype(int)
        data = data_breast_cancer[:,1:]

    #choosing best number of neighbors using accuracy as a criteria
    best_accuracy = 0
    best_k = 0
    best_confusion_matrix = np.empty((2,2))
    accuracy_evolution = np.empty(max_k)
    for k in range(1, max_k+1):
        accuracy, confusion_matrix = try_k(data, k, dataset, 3)
        accuracy_evolution[k-1] = accuracy

        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_k = k
            best_confusion_matrix = confusion_matrix
    

    #displaying results for the best value of k 
    print("the best number of neighbors for dataset "+dataset+" is "+str(best_k)+", accuracy is "+str(best_accuracy*100)+"%")
    if(dataset == 'haberman'):
        plot_confusion_matrix(best_confusion_matrix, ['1','2'])
        try: 
            try_k(data, best_k, dataset, 3, True, coord1, coord2, coord3)
        except ValueError: 
            print("for the haberman dataset, coordinates to display cannot be greater than 2, displaying default coordinates instead")
            try_k(data, best_k, dataset, 3, True)
        
    else:
        plot_confusion_matrix(best_confusion_matrix, ['2','4'])
        try: 
            try_k(data, best_k, dataset, 3, True, coord1, coord2, coord3)
        except ValueError: 
            print("for the haberman dataset, coordinates to display cannot be greater than 8, displaying default coordinates instead")
            try_k(data, best_k, dataset, 3, True)

    plt.figure()
    plt.plot(accuracy_evolution)
    plt.xlabel('number of neighbors')
    plt.ylabel('accuracy')
    plt.xticks(range(0, max_k+1), range(1, max_k+1))

    plt.title('accuracy depending on the number of selected neighbors\n for dataset '+dataset)
    plt.show()




try: 
    search_best_k(20, "haberman", int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
except:
    search_best_k(20, "haberman")
try:    
    search_best_k(20, "breast-cancer", int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
except:
    search_best_k(20, "breast-cancer")

