import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import imutils
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

#########################################CARD-DETECTION MODEL#####################################################
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'
IMAGE_NAME = 'test_images/18110325_1.jpg'

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
def line_bound(line):
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
def horizontal_lines(lines):
    arr_horizontal_lines = []
    for start, line in enumerate(lines, start=0):
        if(line_classfication(line)==1) & (line_bound(line)==0):
            arr_horizontal_lines.append(line)
    #arr_horizontal_lines = np.array(arr_horizontal_lines, dtype=lines.dtype)
    #arr_horizontal_lines = np.sort(arr_horizontal_lines, order=lines[2])
    return arr_horizontal_lines

#Add Vertical Lines in an array
def vertical_lines(lines):
    arr_vertical_lines = []
    for start, line in enumerate(lines, start=0):
        if(line_classfication(line)==0) & (line_bound(line)==0):
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
arr_horizontal_lines_numpy = horizontal_lines(arr_lines_numpy)
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
arr_vertical_lines_numpy = vertical_lines(arr_lines_numpy)
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
k = cv2.waitKey(0)
cv2.destroyAllWindows()

###############################################WARP IMAGE###################################################
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

warped = four_point_transform(image, screenCnt)
print("STEP 3: Apply perspective transform")
cropped = imutils.resize(warped, height=image.shape[0])
cv2.imshow("Scanned", cropped)
cv2.imwrite('output/output.png',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%


lower_blue = np.array([0,0,0])
upper_blue = np.array([255,255,255])
img = cv2.inRange(cropped, lower_blue, upper_blue);
# kernel = (10,10)
# blured = cv2.blur(cropped,kernel)
cv2.imshow("img", img) 
# ret,th3 = cv2.threshold(blured,100,255,cv2.THRESH_BINARY)
# cv2.imshow("th3", th3) 


#%%
# top rgb(10,124,222)
# rgb (2,132,214)
#img = cv2.imread(cropped)
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
lower_hue = np.array([208, 91, 45]) # top rgb(10,124,222)
upper_hue = np.array([203, 98, 42]) # rgb (2,132,214)
mask_gray = cv2.inRange(hsv, lower_hue, upper_hue)
img_res = cv2.bitwise_and(cropped, cropped, mask = mask_gray)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)
cv2.imshow("mask_gray", mask_gray)
cv2.waitKey(0)
cv2.imshow("Filter", img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()

