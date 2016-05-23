
from numpy.linalg import norm
import numpy as np
import cv2

def _line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p2[0]*p1[1] - p1[0]*p2[1])
    return A, B, C


def _segment_intersect(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def _background_color(img):
    """
    Return image background color
    :param img: input image
    :return: background color 0..255
    """
    # treat background color as average color of the edge
    lft, rgt, top, btm = img[0,:], img[-1,:], img[:,0], img[:,-1]
    average = (lft.sum() + rgt.sum() + top.sum() + btm.sum()) / (lft.size + rgt.size + top.size + btm.size)
    return average

def find_chessboard(img):
    """
    Calculate enclosing quad of a chessboard image.
    :param img: image as a numpy array
    :return: 4-element tuple (a, b, c, d) where a, b, c, d is a quad vertices
    """

    """
    Common idea of an algorithm is:
    1) produce binary image from the input where 0s is a background and 255s are belong to chessboard
       btw it can be difficult task for real-word images of a chessboard
    2) find convex hull of contour vertices
    3) approximate resulting set, so nearly parallel segments will collapse into a single segment
    4) get the longest segments and treat them as edges of a chessboard
    5) return resulting segments intersections
    """

    h, w = img.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    img1 = cv2.dilate(img, kernel, 1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverse image if background is lighter than chessboard
    if _background_color(binary_img) >= 128:
        binary_img = 255 - binary_img
    _, contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    vertex_list = np.concatenate(tuple(contours))
    hull = cv2.convexHull(vertex_list)
    poly = cv2.approxPolyDP(hull, 1, True)
    # create segment list annotated with length and index
    seg_list = [(idx, norm(p1 - p2), p1, p2) for (p1, p2, idx) in
                zip(poly, np.roll(poly, -1, axis=0), range(len(poly)))]
    # poly_closed = np.append(poly, [poly[0]], axis=0)
    # take 4 longest segments
    seg_list = sorted(seg_list, key=lambda x: x[1], reverse=True)[:4]
    # revert natural order of segments
    seg_list = sorted(seg_list, key=lambda x: x[0])
    # calculate intersections
    to_line = lambda x: _line(list(x[2].flatten()), list(x[3].flatten()))
    return (tuple([int(v) for v in _segment_intersect(to_line(seg1), to_line(seg2))])
            for seg1, seg2 in zip(seg_list, seg_list[1:] + seg_list[:1]))
