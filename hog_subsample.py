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


# define a single function that can extract features using
# hog sub-sampling and make predictions
def find_cars(p_img, p_ystart, p_ystop, p_scale, p_svc, p_X_scaler, p_orient,
              p_pix_per_cell, p_cell_per_block):
    
    # prepare output
    draw_img = np.copy(p_img)
    
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
            
            # scale features and make a prediction
            test_features = p_X_scaler.transform(hog_features.reshape(1, -1))
            
            test_prediction = p_svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * p_scale)
                ytop_draw = np.int(ytop * p_scale)
                win_draw = np.int(window * p_scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + p_ystart), (
                    xbox_left + win_draw, ytop_draw + win_draw + p_ystart),
                              (0, 0, 255), 6)
            else:
                xbox_left = np.int(xleft * p_scale)
                ytop_draw = np.int(ytop * p_scale)
                win_draw = np.int(window * p_scale)
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + p_ystart), (
                #     xbox_left + win_draw, ytop_draw + win_draw + p_ystart),
                #               (255, 255, 0), 6)
    
    return draw_img


if __name__ == '__main__':
    
    m_ystart = 400
    m_ystop = 656
    m_scale = 1.3
    
    dist_pickle = pickle.load(open("svc.p", "rb"))
    m_svc = dist_pickle["svc"]
    m_X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]

    # load a test image
    img = mpimg.imread('test_images/test1.jpg')

    # find the cars
    out_img = find_cars(img, m_ystart, m_ystop, m_scale, m_svc, m_X_scaler, orient,
                        pix_per_cell, cell_per_block)

    # show the result image
    plt.imshow(out_img)
    
    pass

