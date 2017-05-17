from hog_subsampling import scan_picture, draw_rectangles, draw_labeled_bboxes
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    # load parameters used to train the svm
    with open(r"svc.p", "rb") as pfile:
        pickled = pickle.load(pfile)

    m_svc = pickled["svc"]
    m_X_scaler = pickled["scaler"]
    orient = pickled["orient"]
    pix_per_cell = pickled["pix_per_cell"]
    cell_per_block = pickled["cell_per_block"]
    
    # create the figure
    plt.figure(figsize=(10, 10))

    # load a test image
    img = mpimg.imread('test_images/test1.jpg')

    # rescale!
    # img = img.astype(np.float32) / 255
    
    # find the cars
    m_rects = scan_picture(img, m_svc, m_X_scaler, orient, pix_per_cell,
                           cell_per_block)

    # draw all the rectangles found, for the debug purposes
    res_img = draw_rectangles(img, m_rects)

    # show the result image
    plt.imshow(res_img)

    # add the boxes found with heatmap method
    res_img = draw_labeled_bboxes(res_img, m_rects)

    plt.imshow(res_img)
    
    pass