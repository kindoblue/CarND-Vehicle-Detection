import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from hog_classifier import get_hog_features


def convert_color(p_img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(p_img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(p_img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(p_img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(p_img, cv2.COLOR_RGB2YUV)


def draw_rectangles(p_img, p_lst):
    
    # dont change the original
    cimg = np.copy(p_img)
    
    # cycle on all the rectangles and draw them
    for rect in p_lst:
        top, bottom = rect
        cv2.rectangle(cimg, top, bottom, (255, 255, 0), 5)
   
    return cimg


# define a single function that can extract features using
# hog sub-sampling and make predictions. Taken from lessons and
# adapted
def find_cars(p_img, p_ystart, p_ystop, p_scale, p_svc, p_X_scaler, p_orient,
              p_pix_per_cell, p_cell_per_block, p_debug=False):
    # prepare output
    rects = []
    
    # crop the image to get only the relevant area where cars could be
    img_tosearch = p_img[p_ystart:p_ystop, :, :]
    
    # change color space
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    
    # rescale the image to the searched scale
    if p_scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (
            np.int(imshape[1] / p_scale), np.int(imshape[0] / p_scale)))
    
    # define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - p_cell_per_block + 3
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - p_cell_per_block + 3
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    
    nblocks_per_window = (window // p_pix_per_cell) - p_cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    
    # compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, p_orient, p_pix_per_cell, p_cell_per_block,
                            feature_vec=False)
    hog2 = get_hog_features(ch2, p_orient, p_pix_per_cell, p_cell_per_block,
                            feature_vec=False)
    hog3 = get_hog_features(ch3, p_orient, p_pix_per_cell, p_cell_per_block,
                            feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            
            # extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos * p_pix_per_cell
            ytop = ypos * p_pix_per_cell
            
            # scale features
            test_features = p_X_scaler.transform(hog_features.reshape(1, -1))
            
            # and make a prediction using the classifier
            test_prediction = p_svc.predict(test_features)
            
            # save the current rectangle if the car was found in it
            if test_prediction == 1 or p_debug:
                xbox_left = np.int(xleft * p_scale)
                ytop_draw = np.int(ytop * p_scale)
                win_draw = np.int(window * p_scale)
                top = (xbox_left, ytop_draw + p_ystart)
                bottom = (xbox_left + win_draw, ytop_draw + win_draw + p_ystart)
                rects.append((top, bottom))
    
    return rects


def scan_picture(p_img, p_svc, p_X_scaler, p_orient, p_pix_per_cell,
                 p_cell_per_block):
    # prepare output
    rects = []
    
    # define the windows, for each scale, where to search for cars
    img_size = 64
    windows = [(1.0, 400, 400 + img_size),
               (1.0, 416, 416 + img_size),
               (1.5, 400, 400 + int(img_size * 1.5)),
               (1.5, 432, 432 + int(img_size * 1.5)),
               (2.0, 400, 400 + int(img_size * 2.0)),
               (2.0, 432, 432 + int(img_size * 2.0)),
               (3.5, 400, 400 + int(img_size * 3.5)),
               (3.5, 464, 464 + int(img_size * 3.5))]
    
    # for every windows, start searching for cars
    for w in windows:
        scale, start, stop = w
        out = find_cars(p_img, start, stop, scale, p_svc, p_X_scaler,
                        p_orient, p_pix_per_cell, p_cell_per_block)
        if out:
            rects += out

    # rects contains all the matches in the image
    return rects


if __name__ == '__main__':
    
    dist_pickle = pickle.load(open("svc.p", "rb"))
    m_svc = dist_pickle["svc"]
    m_X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    
    # load a test image
    img = mpimg.imread('test_images/test3.jpg')
    
    # find the cars
    m_rects = scan_picture(img, m_svc, m_X_scaler, orient, pix_per_cell,
                           cell_per_block)

    # draw the rectangles found
    res = draw_rectangles(img, m_rects)
    
    # show the result image
    plt.imshow(res)
    
    pass
