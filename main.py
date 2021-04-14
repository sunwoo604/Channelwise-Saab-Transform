import numpy as np
from tensorflow.keras.datasets import mnist,fashion_mnist
from skimage.util import view_as_windows
from pixelhop import Pixelhop
from skimage.measure import block_reduce
import xgboost as xgb
import warnings, gc
import time


np.random.seed(1)

# Preprocess
N_Train_Reduced = 10000    # 10000
N_Train_Full = 60000     # 50000
N_Test = 10000            # 10000

BS = 2000 # batch size

def select_balanced_subset(images, labels, use_num_images, use_classes):
    '''
    select equal number of images from each classes
    '''
    # Shuffle
    num_total=images.shape[0]
    shuffle_idx=np.random.permutation(num_total)
    images=images[shuffle_idx]
    labels=labels[shuffle_idx]
    num_class=len(use_classes)
    num_per_class=int(use_num_images/num_class)
    selected_images=np.zeros((use_num_images,images.shape[1],images.shape[2],images.shape[3]))
    selected_labels=np.zeros(use_num_images)
    for i in range(num_class):
        # images_in_class=images[labels==i]
        idx=(labels==i)
        index=np.where(idx==True)[0]
        images_in_class=np.zeros((index.shape[0],images.shape[1],images.shape[2],images.shape[3]))
        for j in range(index.shape[0]):
            images_in_class[j]=images[index[j]]
        selected_images[i*num_per_class:(i+1)*num_per_class]=images_in_class[:num_per_class]
        selected_labels[i*num_per_class:(i+1)*num_per_class]=np.ones((num_per_class))*i

    # Shuffle again
    shuffle_idx=np.random.permutation(num_per_class*num_class)
    selected_images=selected_images[shuffle_idx]
    selected_labels=selected_labels[shuffle_idx]
    return selected_images,selected_labels

def Shrink(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']
    # TODO: fill in the rest of max pooling

    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # TODO: fill in the rest of neighborhood construction

    return

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def get_feat(X, num_layers=3):
    output = p2.transform_singleHop(X,layer=0)
    if num_layers>1:
        for i in range(num_layers-1):
            output = p2.transform_singleHop(output, layer=i+1)
    return output


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # ---------- Load MNIST data and split ----------
    (x_train, y_train), (x_test,y_test) = mnist.load_data()


    # -----------Data Preprocessing-----------
    x_train = np.asarray(x_train,dtype='float32')[:,:,:,np.newaxis]
    x_test = np.asarray(x_test,dtype='float32')[:,:,:,np.newaxis]
    y_train = np.asarray(y_train,dtype='int')
    y_test = np.asarray(y_test,dtype='int')

    # if use only 10000 images train pixelhop
    x_train_reduced, y_train_reduced = select_balanced_subset(x_train, y_train, use_num_images=N_Train_Reduced, use_classes=[i for i in range(10)])

    x_train /= 255.0
    x_test /= 255.0


    # -----------Module 1: set PixelHop parameters-----------
    # TODO: fill in this part


    # -----------Module 1: Train PixelHop -----------
    # TODO: fill in this part



    # --------- Module 2: get only Hop 3 feature for both training set and testing set -----------
    # you can get feature "batch wise" and concatenate them if your memory is restricted
    # TODO: fill in this part
    
    

    # --------- Module 2: standardization
    STD = np.std(train_hop3_feats, axis=0, keepdims=1)
    train_hop3_feats = train_hop3_feats/STD
    test_hop3_feats = test_hop3_feats/STD
    
    #---------- Module 3: Train XGBoost classifier on hop3 feature ---------

    tr_acc = []
    te_acc = []
    
    clf = xgb.XGBClassifier(n_jobs=-1,
                        objective='multi:softprob',
                        # tree_method='gpu_hist', gpu_id=None,
                        max_depth=6,n_estimators=100,
                        min_child_weight=5,gamma=5,
                        subsample=0.8,learning_rate=0.1,
                        nthread=8,colsample_bytree=1.0)

    # TODO: fill in the rest and report accuracy

    
    
    
    
    
    
    
    
    
