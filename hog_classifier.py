from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# taken from the lessons, it returns HOG features and visualization
def get_hog_features(img,
                     orient,
                     pix_per_cell,
                     cell_per_block,
                     vis=False,
                     feature_vec=True):
    """
    Returns HOG features and, if requested, visualization of the HOG
    ----------
    :param img : (M, N) ndarray
        Input image (greyscale).
    :param orient : int
        Number of orientation bins.
    :param pix_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    :param cell_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    :param vis : bool, optional
        Also return an image of the HOG.
    :param feature_vec : bool, optional
        Return the data as a feature vector by calling .ravel() on the result
        just before returning.
    """
    
    if vis:
        
        # use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(
                                      cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=feature_vec)
        
        return features, hog_image
    
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        
        return features


# taken from the lessons
def extract_features(imgs,
                     cspace='RGB',
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0):
    """
        Returns a list of feature vectors, built from the images
        using their HOG's
        ----------
        :param imgs : list
            Input images
        :param cspace : string
            Color space to consider
        :param orient : int
            Number of orientation bins, for the HOG
        :param pix_per_cell : 2 tuple (int, int)
            Size (in pixels) of a cell, for the HOG
        :param cell_per_block  : 2 tuple (int,int)
            Number of cells in each block, for the HOG
        :param hog_channel : int or string, optional
            Which channel to use. if 'ALL', the HOG's is built on all the
            channels and then the feature vectors packed together
    """
    
    # create a list to append feature vectors to
    features = []
    
    # iterate through the list of images
    for file in imgs:
        
        # read in each one by one
        image = mpimg.imread(file)
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        
        # call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            # cycle on all the channels
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_vect = get_hog_features(feature_image[:, :, channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)
                hog_features.append(hog_vect)
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)
        
        # append the new feature vector to the features list
        features.append(hog_features)
    
    # return list of feature vectors
    return features


def train():
    
    b = glob.glob('images/non-vehicles_smallset/**/*.jpeg', recursive=True)
    a = glob.glob('images/vehicles_smallset/**/*.jpeg', recursive=True)
    
    cars = []
    notcars = []
    
    for image in a:
        cars.append(image)
    
    for image in b:
        notcars.append(image)
    
    # parameters to setup feature extraction
    colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    
    t = time.time()
    car_features = extract_features(cars,
                                    cspace=colorspace,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
    
    notcar_features = extract_features(notcars,
                                       cspace=colorspace,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')

    # create an array stack of feature vectors
    # rows are the samples, columns the features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # fit a per-column scaler
    scaler = StandardScaler().fit(X)
    
    # apply the scaler to X
    scaled_X = scaler.transform(X)

    # define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # use SVC
    svc = SVC()
    
    # parameters to explore
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 3, 4, 5]}
    
    # classifier wrapped by the grid search
    clf = GridSearchCV(svc, parameters)
    
    # check the training time for the SVC
    t = time.time()

    # fit on the the entire dataset, as the searcher will do cross
    # validation on his own
    clf.fit(scaled_X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to discover parameters...')
    print('Best params ', clf.best_params_)
    
    # re-create the classifier with the best parameters
    svc = SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'])
    
    # train
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    
    # check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

if __name__ == '__main__':
    train()
