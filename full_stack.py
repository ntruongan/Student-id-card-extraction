import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import imutils
import predict
import glob
# from PIL import 
from PIL import ImageFont, ImageDraw, Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util



#Find Lines Intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0, 0;

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

#check if line is Horizontal Lines or Vertical Lines
def line_classfication(line):
    if(abs(line[0])<np.cos(45)):
        return 1 #horizontal
    return 0

#Check if Line is at the edge of the Image
def line_bound(line, img):
    if(int(line[1])==int(line[3]) & int(line[3])==0) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == 0 + 1) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == 0 + 2) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == 0 + 3) | \
            (int(line[1])==int(line[3]) & int(line[3]) == int(img.shape[1])) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == int(img.shape[1])-1) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == int(img.shape[1]) - 2) | \
            (int(line[1]) == int(line[3]) & int(line[3]) == int(img.shape[1]) - 3) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == 0) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == 0 + 1) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == 0 + 2) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == 0 + 3) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == int(img.shape[0])) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == int(img.shape[0]) - 1) | \
            (int(line[2]) == int(line[4]) & int(line[2]) == int(img.shape[0]) - 2)| \
            (int(line[2]) == int(line[4]) & int(line[2]) == int(img.shape[0]) - 3):
            return 1
    return 0

#After Checking, add Horizontal Lines in an array
def horizontal_lines(lines, img):
    arr_horizontal_lines = []
    for start, line in enumerate(lines, start=0):
        if(line_classfication(line)==1) & (line_bound(line, img)==0):
            arr_horizontal_lines.append(line)
    #arr_horizontal_lines = np.array(arr_horizontal_lines, dtype=lines.dtype)
    #arr_horizontal_lines = np.sort(arr_horizontal_lines, order=lines[2])
    return arr_horizontal_lines

#Add Vertical Lines in an array
def vertical_lines(lines, img):
    arr_vertical_lines = []
    for start, line in enumerate(lines, start=0):
        if(line_classfication(line)==0) & (line_bound(line, img)==0):
            arr_vertical_lines.append(line)
    #arr_vertical_lines = np.array(arr_vertical_lines,dtype=lines.dtype)
    #arr_vertical_lines = np.sort(arr_vertical_lines,order='x1')
    return arr_vertical_lines

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = pts
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped



#########################################CARD-DETECTION MODEL#####################################################
# Name of the directory containing the object detection module we're using
def id_card_align(image):
    MODEL_NAME = 'model'
    IMAGE_NAME = 'test_images/%s'%image
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')
    
    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
    
    # Number of classes the object detector can identify
    NUM_CLASSES = 1
    
    ########################################CARD DETECTION AND CROP##################################################
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
    
    # Define input and output tensors (i.e. data) for the object detection classifier
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    a = detection_graph.get_all_collection_keys()
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)
    
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    #print(boxes)
    #print(scores)
    
    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)
    
    ymin, xmin, ymax, xmax = array_coord
    print(array_coord)
    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = ((xmin-0.012) * im_width, (xmax+0.012) * im_width, (ymin-0.012) * im_height, (ymax+0.012) * im_height)
    
    image_path=IMAGE_NAME
    #image_path = "images/18110349_0.jpg"
    output_path="test_images/output_1.jpg"
    # Using Image to crop and save the extracted copied image
    im = Image.open(image_path)
    im.crop((left, top, right, bottom)).save(output_path, quality=95)
    
    
    cv2.imshow('ID-CARD-DETECTOR : ', image)
    cv2.waitKey(0)
    
    image_cropped = cv2.imread(output_path)
    cv2.imshow("ID-CARD-CROPPED : ", image_cropped)
    cv2.waitKey(0)
    
    #############################################IMAGE PROCESSING####################################################
    image = image_cropped
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    grey_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur_grey = cv2.medianBlur(grey_img,5)
    canny_grey = cv2.Canny(blur_grey.copy(),10,250)
    ret, threshInv1 = cv2.threshold(blur_grey.copy(), 160, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold1", threshInv1)
    
    #canny1 = cv2.Canny(threshInv1,10,220,apertureSize=3)
    ret, threshInv2 = cv2.threshold(blur_grey.copy(), 135, 255, cv2.THRESH_BINARY)
    #canny2 = cv2.Canny(threshInv2,10,220,apertureSize=3)
    cv2.imshow("Threshold2", threshInv2)
    grab = cv2.bitwise_and(threshInv1,threshInv2)
    
    contours, hierarchy = cv2.findContours(grab.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(grab, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
    # see the results
    cv2.imshow('EXTERNAL', grab)
    print(f"EXTERNAL: {hierarchy}")
    cv2.waitKey(0)
    
    #grab_canny = cv2.bitwise_or(canny1,canny2)
    #cv2.imwrite('can_grab_out.jpg',grab_canny)
    #grab = threshInv1
    cv2.imshow("Threshold Grabbed", grab)
    cv2.imwrite('thre_grab_out.jpg',grab)
    cv2.waitKey(0)
    #############################################FIND LINES AND POINTS###############################################
    
    
    img = cv2.imread('thre_grab_out.jpg')
    #img2 = cv2.imread('can_grab_out.jpg')
    img_after = cv2.imread('thre_grab_out.jpg')
    #img = grab.copy()
    blur = cv2.medianBlur(img,7)
    #blur = cv2.GaussianBlur(img,(7,9),cv2.BORDER_DEFAULT)
    #blur2 = cv2.GaussianBlur(img2,(7,7),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blur, 50, 150,apertureSize=7,L2gradient=True)
    #edges2 = cv2.Canny(img2, 20, 220, apertureSize=3)
    edges_grab = cv2.bitwise_or(edges, canny_grey)
    #edges = cv2.Canny(blur, 8, 200,apertureSize=3)
    cv2.imshow('edges', edges_grab)
    cv2.waitKey(0)
    lines = cv2.HoughLines(edges_grab.copy(), 1, np.pi/180, 182)
    
    #Line Datatype where First Point(x1,x2), Second Point(x3,x4)
    dtype = [('a', 'float'), ('x1', 'float'), ('x2', 'float'), ('x3', 'float'), ('x4', 'float'), ('rho', 'float')]
    
    # array save parameter to calculate late
    arr_lines = []
    
    for line in lines:
        rho, theta = line[0]
    
        a = np.cos(theta)
        b = np.sin(theta)
    
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        x1 = int(x0 + int(img.shape[1]) * (-b))
        # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        y1 = int(y0 + int(img.shape[1]) * (a))
        # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        x2 = int(x0 - int(img.shape[1]) * (-b))
        # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        y2 = int(y0 - int(img.shape[1]) * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        arr_lines.append((a, x1, y1, x2, y2, rho))
    
    #Find Lines
    arr_lines_numpy = np.array(arr_lines, dtype=dtype)
    arr_lines_numpy = np.sort(arr_lines_numpy, order='a')
    
    #Find Horizontal Lines
    arr_horizontal_lines_numpy = horizontal_lines(arr_lines_numpy, img)
    arr_horizontal_lines_numpy = np.array(arr_horizontal_lines_numpy, dtype=dtype)
    arr_horizontal_lines_numpy = np.sort(arr_horizontal_lines_numpy, order=dtype[4][0])
    print(img.shape[0])
    print(len(arr_horizontal_lines_numpy))
    print(arr_horizontal_lines_numpy[0])
    
    
    #Draw First and Last Horizontal Line
    hori_last = len(arr_horizontal_lines_numpy) - 1
    cv2.line(img,(int(arr_horizontal_lines_numpy[0][1]),int(arr_horizontal_lines_numpy[0][2])),(int(arr_horizontal_lines_numpy[0][3]),int(arr_horizontal_lines_numpy[0][4])),(0,255,0),2)
    cv2.line(img,(int(arr_horizontal_lines_numpy[hori_last][1]),int(arr_horizontal_lines_numpy[hori_last][2])),(int(arr_horizontal_lines_numpy[hori_last][3]),int(arr_horizontal_lines_numpy[hori_last][4])),(0,255,0),2)
    
    #Find Vertical Lines
    first_hori = arr_horizontal_lines_numpy[0]
    arr_vertical_lines_numpy = vertical_lines(arr_lines_numpy, img)
    if (len(arr_vertical_lines_numpy)==0):
        arr_vertical_lines_numpy.append((-0.8,int(img.shape[1]/8),int(img.shape[0]/8),int(img.shape[1]/8),int(img.shape[0]-img.shape[0]/8),first_hori[5]+90))
    arr_vertical_lines_numpy = np.array(arr_vertical_lines_numpy, dtype=dtype)
    arr_vertical_lines_numpy = np.sort(arr_vertical_lines_numpy, order=dtype[1][0])
    print(img.shape[1])
    print(len(arr_vertical_lines_numpy))
    #print(arr_vertical_lines_numpy[0])
    
    #Draw First and Last Vertical Line
    verti_last = len(arr_vertical_lines_numpy) - 1
    cv2.line(img,(int(arr_vertical_lines_numpy[0][1]),int(arr_vertical_lines_numpy[0][2])),(int(arr_vertical_lines_numpy[0][3]),int(arr_vertical_lines_numpy[0][4])),(0,255,0),2)
    cv2.line(img,(int(arr_vertical_lines_numpy[verti_last][1]),int(arr_vertical_lines_numpy[verti_last][2])),(int(arr_vertical_lines_numpy[verti_last][3]),int(arr_vertical_lines_numpy[verti_last][4])),(0,255,0),2)
    
    #Array of Four Points
    screenCnt = np.zeros((4, 2), dtype="float32")
    
    #Top Left Point
    lineA = arr_vertical_lines_numpy[0]
    lineB = arr_horizontal_lines_numpy[0]
    x1,y1 = line_intersection(([round(lineA[1]), round(lineA[2])], [round(lineA[3]), round(lineA[4])]),
                                           ([round(lineB[1]), round(lineB[2])], [round(lineB[3]), round(lineB[4])]))
    
    
    #Top Right Point
    lineA1 = arr_vertical_lines_numpy[verti_last]
    lineB1 = arr_horizontal_lines_numpy[0]
    x2,y2 = line_intersection(([round(lineA1[1]), round(lineA1[2])], [round(lineA1[3]), round(lineA1[4])]),
                                           ([round(lineB1[1]), round(lineB1[2])], [round(lineB1[3]), round(lineB1[4])]))
    
    
    #Bottom Left Point
    lineA2 = arr_vertical_lines_numpy[0]
    lineB2 = arr_horizontal_lines_numpy[hori_last]
    x3,y3 = line_intersection(([round(lineA2[1]), round(lineA2[2])], [round(lineA2[3]), round(lineA2[4])]),
                                           ([round(lineB2[1]), round(lineB2[2])], [round(lineB2[3]), round(lineB2[4])]))
    
    
    #Bottom Right Point
    lineA3 = arr_vertical_lines_numpy[verti_last]
    lineB3 = arr_horizontal_lines_numpy[hori_last]
    x4,y4 = line_intersection(([round(lineA3[1]), round(lineA3[2])], [round(lineA3[3]), round(lineA3[4])]),
                                           ([round(lineB3[1]), round(lineB3[2])], [round(lineB3[3]), round(lineB3[4])]))
    
    #TL, TR, BR, BL
    screenCnt[0]=[x1, y1]
    screenCnt[1]=[x2, y2]
    screenCnt[2]=[x4, y4]
    screenCnt[3]=[x3, y3]
    print(screenCnt)
    
    cv2.circle(img, (int(round(x1)), int(round(y1))), 25, (0, 255, 0), -1)
    cv2.circle(img, (int(round(x2)), int(round(y2))), 25, (0, 255, 0), -1)
    cv2.circle(img, (int(round(x3)), int(round(y3))), 25, (0, 255, 0), -1)
    cv2.circle(img, (int(round(x4)), int(round(y4))), 25, (0, 255, 0), -1)
    cv2.imshow('Detect', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    ###############################################WARP IMAGE###################################################
    
    warped = four_point_transform(image, screenCnt)
    print("STEP 3: Apply perspective transform")
    cropped = imutils.resize(warped, height=image.shape[0])
    cv2.imshow("Scanned", cropped)
    cv2.imwrite('output/output.png',cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped


#%%

def blue_area_detec(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,120, 0])
    upper_blue = np.array([50, 255,110])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imwrite('1_origin.png', img)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('2_gray.png', img_grey)
    ret, thresh_img = cv2.threshold(img_grey, 65, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('3_thresh_img.png', thresh_img)
    blue_area = cv2.bitwise_and(img, img, mask = mask)
    blue_area = cv2.cvtColor(blue_area, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('3_blue_area.png', blue_area)
    # blue_area_contours, _ = cv2.findContours(blue_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    blue_area_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
     
    for idx, cnt in enumerate(blue_area_contours): 
        x, y, w, h = cv2.boundingRect(cnt) 
        if w*h>100:
            thresh_img[ y:y+h, x:x+w]=0
            thresh_img = cv2.drawContours(thresh_img,blue_area_contours, idx, (0, 255, 244), 3)
            print(idx,x, y, w, h)
    return thresh_img


    

def crop(name='18110325_1.jpg'):
    # path_image_test = r"D:\IT\Nam_3\HK2\DL\Final_Project\object_detection\object_detection\output\output.png"
    img = id_card_align(name)
    img = cv2.resize(img, (800,500))
    
     
    # img = cv2.imread(path_image_test)
    kernel = np.ones((3,30),np.uint8)
    
    thresh_img = blue_area_detec(img)
    cv2.imwrite('4_thresh_img.png', thresh_img)
    thresh_img = cv2.dilate(thresh_img, kernel,iterations = 1)
    cv2.imwrite('5_dilate.png', thresh_img)
    thresh_contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
     	cv2.CHAIN_APPROX_SIMPLE)
    
    our_contours = []
    for idx, cnt in enumerate(thresh_contours): 
        x, y, w, h = cv2.boundingRect(cnt) 
        if w*h>2000 and w*h<=10000 and w<700 and h <100:
            # thresh_img[ y:y+h, x:x+w]=0
            our_contours.append(cnt)
            # crop = thresh_contours[ y:y+h, x:x+w]
            cv2.imwrite('crop%d.png'%idx,img[ y:y+h, x:x+w])
            print(idx,x, y,w,h, w* h)
        else :
            thresh_img[ y:y+h, x:x+w]=0

    cv2.imshow("img.png", img) 
    cv2.imwrite('my_last.png', img)
    # cv2.imshow("Filter", blue_area)
    # cv2.imwrite('7_blue_area.png', blue_area)
    # cv2.imshow("thr", thresh_img)
    # cv2.imwrite('8_thresh_img_after_remove_6.png', thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# crop('18110309_4.jpg')
# crop('18110332_2.JPG')   

# crop('18124123_1.jpg  ')  

# crop(r'18161212_1.jpg')
 



def kmean(img, K, white_background = False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    count = sum(label) 
    while_color = np.array([[0,0,0],[255,255,255]], dtype = 'uint8')
    black_color = np.array([[255,255,255],[0,0,0]], dtype = 'uint8')
    if count >= len(label)/2:
        if white_background:
            center = while_color 
        else:
            center = black_color
    else:
        if white_background:
            center = black_color 
        else:
            center = while_color
        
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2


def get_neighbor_contour(cnt_nei, cnt_home):
    print('Finding contours neighbors....')
    x_home, y_home, w_home, h_home = cnt_home
    cnt_neighbors = cnt_nei
    cnt_neighbors.remove(cnt_home)
    n = len(cnt_neighbors)
    for i in range(0,n):
        neighbors = []
        for cnt in cnt_neighbors:
            x_cnt, y_cnt, w_cnt, h_cnt = cnt
            if is_neighbors(x_home, y_home, w_home, h_home, x_cnt, y_cnt, w_cnt, h_cnt):
                x_home, y_home, w_home, h_home = merge_cnt(x_home, y_home, w_home, h_home, x_cnt, y_cnt, w_cnt, h_cnt)
                # cnt_neighbors.remove(cnt)
                neighbors.append(cnt)
                
        for nei in neighbors:
            cnt_neighbors.remove(nei)
    return x_home, y_home, w_home, h_home
            
    
    # pass
    

def merge_cnt(x1, y1, w1, h1, x2, y2, w2, h2):
    # x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    # x2, y2, w2, h2 = cv2.boundingRect(cnt2)
    print('\nmerging: \n\t{x1: %d, y1: %d, w1: %d, h1: %d} and \n\t{x2: %d, y2: %d, w2: %d, h2: %d}'%( x1, y1, w1, h1, x2, y2, w2, h2))
    x_merge = min(x1,x2)
    y_merge = min(y1,y2)
    w_merge = w1+w2+5
    h_merge = max(h1,h2)
    print('into: \n\t{x_merge: %d, y_merge: %d, w_merge: %d, h_merge: %d \n}'%(x_merge, y_merge, w_merge, h_merge))
    return (x_merge, y_merge, w_merge, h_merge)

def is_neighbors(x_home, y_home, w_home, h_home, x_cnt, y_cnt, w_cnt, h_cnt):
    distance = 0
    # print("comparing",x_home, y_home, w_home, h_home,'and', x_cnt, y_cnt, w_cnt, h_cnt)
    # x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    # x2, y2, w2, h2 = cv2.boundingRect(cnt2)
    if x_home>x_cnt:
        x_dist = abs(x_cnt+w_cnt-x_home)
    else:
        x_dist = abs(x_home+w_home-x_cnt)
        
    distance = abs(y_home-y_cnt) + x_dist + abs(h_home-h_cnt) 
    # print(distance)
    # print(distance<30)
    return distance<50
    
    
    # return distance
    # pass
    

def get_text_location(img):
    img = img
    img = cv2.resize(img, (800,500))
    img_kmean = kmean(img,2)
    cv2.imwrite('image_processing/kmean.jpg',img_kmean)
    res2_gray = cv2.cvtColor(img_kmean, cv2.COLOR_RGB2GRAY)
    cv2.imshow('COLOR_RGB2GRAY', res2_gray)
    
    kernel = np.ones((1,8),np.uint8)
    
    dilate = cv2.dilate(res2_gray,kernel,iterations = 1)
    cv2.imshow('res2_dilate', dilate)
    cv2.imwrite('image_processing/kmean_dilate.jpg',dilate)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
     	cv2.CHAIN_APPROX_SIMPLE)
    
    mycrop = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        if (x+x+w)/2>350 and (y+y+h)/2>150 and w<400 and h<100 and (y+y+h)/2 < 350 and w*h > 200:
            print("x: %d; y: %d; w: %d; h: %d"%(x,y,w,h))
            cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            # res2[x:x+w][y:y+h]=(0,0,0)  
            # cropped = res2[y:y + h, x:x + w] 
            mycrop.append((x, y, w, h))
        else:
            if x < 250 and y > 430 and h < 100 and w < 400 and w*h > 350 and h > 10 and x > 10:
            # if x < 80 and y > 400: 
                cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (255, 255, 0), 2)
                print("x: %d; y: %d; w: %d; h: %d"%(x,y,w,h)) 
                mycrop.append((x, y, w, h))
            # else:
                # if w<100 and h <100:
                # res2[y:y + h, x:x + w]  = 0
                # else:
                    
     
    cv2.imshow('img', img)
    cv2.imwrite('image_processing/contours.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mycrop
        

def merge_contours(mycrop):
    neigh_found = []
    print('MERGING...')
    while len(mycrop) > 0:
        for drop in mycrop: 
            print("NUMBER OF NEIGHBOR:", len(mycrop)) 
            neigh_found.append(get_neighbor_contour(mycrop, drop))
            
    return neigh_found


def clear_folder(folder_name):
    names = glob.glob(folder_name+'/*')
    for name in names:
        path = os.getcwd()+'/'+name
        if os.path.exists(path):
            os.remove(path)
        else:
            print("The file %s does not exist"%path)
        # os.
    # os.getcwd()



if __name__=='__main__':
    img = id_card_align(r'output_1.jpg')
    img = cv2.resize(img, (800,500))
    
    image_text_location = []
    image_text_location = get_text_location(img)
    
    image_text_location.sort()
    neigh_found = merge_contours(image_text_location) 
    
    img_text = []
    print('mydrop',len(image_text_location))
    for nei in neigh_found:
        img_temp = img[nei[1]-5:5+nei[1]+nei[3],nei[0]-5:5+nei[0]+nei[2]]
        img_text.append(img_temp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    paths = []
    mapping_loc_name = {}
    clear_folder('text_images')
    for idx, crop_ in enumerate (img_text):
        z = kmean(crop_, 2, white_background=True)
        # mapping_loc_name['idx'] = path
        path = 'text_images/crop%d%d%d%d.jpg'%(neigh_found[idx])
        mapping_loc_name[idx] = 'crop%d%d%d%d.jpg'%(neigh_found[idx])
        paths.append(path)
        cv2.imwrite(path,z)      
        
    text_diction = predict.predict(model = 'crnn_model', datapath = 'text_images/')
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    fontpath = "./Times New Roman Ecofont.ttf" 
    font = ImageFont.truetype(fontpath, 25)
    
    
    for idx, nei in enumerate (neigh_found):
        label = 'crop%d%d%d%d.jpg'%(neigh_found[idx])
        text = text_diction[label] 
        


        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        b,g,r,a = 0,0,255,0
        draw.text((nei[0], nei[1]-20),  text, font = font, fill = (b, g, r, a))
        img = np.array(img_pil)
        
        
        # cv2.putText(img, text,(nei[0], nei[1]), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (nei[0], nei[1]), (nei[0]+nei[2], nei[1]+nei[3]), (0, 255, 0), 2)

    cv2.imwrite('output/result.jpg',img)
    cv2.imshow('result',img) 
    

