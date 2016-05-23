import argparse
import os
from azoft.img_utils import *


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input image")
    parser.add_argument("-s", "--show", help="show resulting quad", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("error: image is not found (%s)"%args.image)
        exit(1)

    input_img = cv2.imread(args.image)
    pts = list(find_chessboard(input_img))
    for p, seg in zip(pts, zip(pts, pts[1:] + pts[:1])):
        print(p)
        if args.show:
            input_img = cv2.line(input_img, seg[0], seg[1], color=(0,0,255), lineType=cv2.LINE_AA)
            input_img = cv2.circle(input_img, p, 3, (0,255,0), lineType=cv2.LINE_AA)
    if args.show:
        cv2.imshow("result", input_img)
        0xFF & cv2.waitKey()
        cv2.destroyAllWindows()
