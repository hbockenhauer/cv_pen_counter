import scipy
import skimage
import skimage.transform
import skimage.util
import numpy as np
import os, glob

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict

import glob, os, re

def list_images(image_dir, filename_expression='*.jpg'):
    # Convert the filename expression to a regex pattern, replacing '*' with '.*' and adding start (^) and end ($) anchors
    filename_expression_regex = re.compile('^' + filename_expression.replace('*', '.*') + '$', re.IGNORECASE)
    # List all files in the directory
    all_files = glob.glob(os.path.join(image_dir, '*'))
    # Filter files using the regex pattern to match case-insensitively
    filenames = [f for f in all_files if filename_expression_regex.match(os.path.basename(f))]
    filenames = sorted(filenames)  # Important for cross-platform compatibility
    print(f'Found {len(filenames)} image files in the directory "{image_dir}"')
    return filenames

IMAGE_DIR = 'images/lab_partner1'

# list all images. There should be 37 images in the images/mypen/ directory
filenames = list_images(IMAGE_DIR)
N = len(filenames)

def get_image_width(I):
    return I.shape[1]

def get_image_height(I):
    return I.shape[0]
    
def get_image_channels(I):
    return I.shape[2]

Is = [plt.imread(filename) for filename in filenames]
print('loaded %d images' % len(Is))

annots = None # store your solution in this variable name
annot_filename = os.path.join(IMAGE_DIR, 'annots.npy')

annots = np.load(annot_filename, allow_pickle= True)
N = len(Is)

train_size = int(0.75 * N)

train_imgs = list(range(0, train_size))
test_imgs = list(range(train_size, N))


def show_annotation(I, p1, p2):
    plt.figure()
    print(p1,p2)
    plt.imshow(I)
    plt.plot(p1[0], p1[1], marker = 'o', color = 'green', markersize = '10', label = 'tip')
    plt.plot(p2[0], p2[1], marker = 'o', color = 'blue', markersize = '10', label = 'end')
    plt.plot([p1[0], p2[0]], [p1[1] ,p2[1]], linestyle = '-', color = 'yellow', linewidth = '2')
    plt.legend()
    
    # done, show the image
    plt.show()

img_idx = 0
I = Is[img_idx]
p1 = annots[img_idx,:2].copy() # point 1, tip of the pen
p2 = annots[img_idx,2:].copy() # point 2, end of the pen

show_annotation(I, p1, p2)

def show_annotated_image(j, train_imgs, Is, annots):
    # Show the j-th training image
    img_idx = train_imgs[j]
    
    I = Is[img_idx]
    p1 = annots[img_idx, :2].copy()  # point 1, tip of the pen
    p2 = annots[img_idx, 2:].copy()  # point 2, end of the pen

    show_annotation(I, p1, p2)

# for j in range(len(train_imgs)):
#     show_annotated_image(j, train_imgs, Is, annots)

# the size of the patch in pixels
WIN_SIZE = (100, 100, 3)

# for convenience, half the window
HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])

def sample_points_grid(I):
    # window centers
    W = get_image_width(I)
    H = get_image_height(I)
    
    step_size = (WIN_SIZE[0]//2, WIN_SIZE[1]//2)
    min_ys = range(0, H-WIN_SIZE[0]+1, step_size[0])
    min_xs = range(0, W-WIN_SIZE[1]+1, step_size[1])
    center_ys = range(HALF_WIN_SIZE[0], H-HALF_WIN_SIZE[0]+1, step_size[0])
    center_xs = range(HALF_WIN_SIZE[1], W-HALF_WIN_SIZE[1]+1, step_size[1])
    centers = np.array(np.meshgrid(center_xs, center_ys))
    centers = centers.reshape(2,-1).T
    centers = centers.astype(float) 
    
    # add a bit of random offset
    centers += np.random.rand(*centers.shape) * 10 
    
    # discard points close to border where we can't extract patches
    centers = remove_points_near_border(I, centers)
    
    return centers

def sample_points_around_pen(I, p1, p2):
    Nu = 100 # uniform samples (will mostly be background, and some non-background)
    Nt = 50 # samples at target locations, i.e. near start, end, and middle of pen
    
    target_std_dev = np.array(HALF_WIN_SIZE[:2])/3 # variance to add to locations

    upoints = sample_points_grid(I)
    idxs = np.random.choice(upoints.shape[0], Nu)
    upoints = upoints[idxs,:]
    
    
    # sample around target locations
    tpoints1 = np.random.randn(Nt,2)
    tpoints1 = tpoints1 * target_std_dev + p1

    tpoints2 = np.random.randn(Nt,2)
    tpoints2 = tpoints2 * target_std_dev + p2

    # sample over length pen
    alpha = np.random.rand(Nt)
    tpoints3 = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
    tpoints3 = tpoints3 + np.random.randn(Nt,2) * target_std_dev
    
    # merge all points
    points = np.vstack((upoints, tpoints1, tpoints2, tpoints3))
    
    # discard points close to border where we can't extract patches
    points = remove_points_near_border(I, points)
    
    return points

def remove_points_near_border(I, points):
    W = get_image_width(I)
    H = get_image_height(I)

    # discard points that are too close to border
    points = points[points[:,0] > HALF_WIN_SIZE[1],:]
    points = points[points[:,1] > HALF_WIN_SIZE[0],:]
    points = points[points[:,0] < W - HALF_WIN_SIZE[1],:]
    points = points[points[:,1] < H - HALF_WIN_SIZE[0],:]
    
    return points

points1 = sample_points_grid(I) # sampling strategy 1
points2 = sample_points_around_pen(I, p1, p2) # sampling strategy 2

# plot both sampling strategies in a single figure using subplots
plt.figure(figsize=(10,12))
plt.subplot(1,2,1)
plt.imshow(I)
plt.plot(points1[:,0], points1[:,1], 'w.')
plt.title('sampling strategy 1')

plt.subplot(1,2,2)
plt.imshow(I)
plt.plot(points2[:,0], points2[:,1], 'w.')
plt.title('sampling strategy 2')
plt.show()

def get_patch_at_point(I, p):
    xmax = int(p[0])+50
    xmin = int(p[0])-50
    ymax = int(p[1])+50
    ymin = int(p[1])-50
    #print(xmin,xmax,ymin,ymax)
    P = I[ymin:ymax, xmin: xmax]
    return P

P = get_patch_at_point(I, p1)
plt.imshow(P)
plt.show()

CLASS_NAMES = [
    'background', # class 0
    'tip',        # class 1
    'end',        # class 2
    'middle'      # class 3
]

def make_labels_for_points(I, p1, p2, points):
    """ Determine the class label (as an integer) on point distance to different parts of the pen """
    num_points = points.shape[0]
    
    # for all points ....
    
    # ... determine their distance to tip of the pen
    dist1 = points - p1
    dist1 = np.sqrt(np.sum(dist1 * dist1, axis=1))
    
    # ... determine their distance to end of the pen
    dist2 = points - p2
    dist2 = np.sqrt(np.sum(dist2 * dist2, axis=1))

    # ... determine distance to pen middle
    alpha = np.linspace(0.2, 0.8, 100)
    midpoints = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None]) 
    dist3 = scipy.spatial.distance_matrix(midpoints, points)
    dist3 = np.min(dist3, axis=0)
    
    # the class label of a point will be determined by which distance is smallest
    #    and if that distance is at least below `dist_thresh`, otherwise it is background
    dist_thresh = WIN_SIZE[0] * 2./3.

    # store distance to closest point in each class in columns
    class_dist = np.zeros((num_points, 4))
    class_dist[:,0] = dist_thresh
    class_dist[:,1] = dist1
    class_dist[:,2] = dist2
    class_dist[:,3] = dist3
    
    # the class label is now the column with the lowest number
    labels = np.argmin(class_dist, axis=1)
    
    return labels

def plot_labeled_points(points, labels):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'r.', label=CLASS_NAMES[0])
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'g.', label=CLASS_NAMES[1])
    plt.plot(points[labels == 2, 0], points[labels == 2, 1], 'b.', label=CLASS_NAMES[2])
    plt.plot(points[labels == 3, 0], points[labels == 3, 1], 'y.', label=CLASS_NAMES[3])

labels1 = make_labels_for_points(I, p1, p2, points1)
labels2 = make_labels_for_points(I, p1, p2, points2)

plt.figure(figsize=(10,12))

plt.subplot(1,2,1)
plt.imshow(I)
plot_labeled_points(points1, labels1)
plt.legend()

plt.subplot(1,2,2)
plt.imshow(I)
plot_labeled_points(points2, labels2)
plt.legend()

plt.show() 