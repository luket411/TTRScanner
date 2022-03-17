from sys import path
from os import listdir, path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import numpy as np
import cv2
import matplotlib.pyplot as plt
from board_handling.warp_board import annotate_fixed_city_points
from scripts.get_and_save_corners import getImagesInDir

def orb_get_keypoints_descriptors(img, draw=False):   

    orb = cv2.ORB_create()

    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    if draw:
        drawn_image = np.copy(img)
        drawn_image = cv2.drawKeypoints(img, keypoints, drawn_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(drawn_image)
        plt.show()
    
    return keypoints, descriptors

def sift_get_keypoints_descriptors(img, draw=False):   

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if draw:
        drawn_image = np.copy(img)
        drawn_image = cv2.drawKeypoints(img, keypoints, drawn_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(drawn_image)
        plt.show()
    
    return keypoints, descriptors

def draw_matches(possiblyCorrectMatches, source_image, source_keypoints, target_image, target_keypoints):
    img_matched = cv2.drawMatchesKnn(source_image, source_keypoints, target_image, target_keypoints, possiblyCorrectMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig=plt.figure(figsize=(24,24)) 
    plt.imshow(img_matched)
    plt.show()

def main(source_file = "assets/3.1 Blue-Yellow,Green,Gray/PXL_20220209_153108922.jpg", target_file = "assets/0.0 Cropped/11.png"):
    source_img = cv2.imread(source_file, 1)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.imread(target_file, 1)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    return find_homography_between_images(source_img, target_img)

def find_homography_between_images(source_img, target_img):
    source_keypoints, source_descriptors = orb_get_keypoints_descriptors(source_img)
    target_keypoints, target_descriptors = orb_get_keypoints_descriptors(target_img)
    
    bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck = False)
    neighbours = 2
    matches = bf.knnMatch(source_descriptors, target_descriptors, neighbours)
    possiblyCorrectMatches = []
    targetpts = []
    sourcepts = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            possiblyCorrectMatches.append([m])
            sourcepts.append(source_keypoints[m.queryIdx].pt)
            targetpts.append(target_keypoints[m.trainIdx].pt)
    
    possiblyCorrectMatches = np.array(possiblyCorrectMatches)
    targetpts = np.array(targetpts, dtype=np.float32)
    sourcepts = np.array(sourcepts, dtype=np.float32)
    
    
    np.random.shuffle(possiblyCorrectMatches)
    
    H = cv2.findHomography(sourcepts, targetpts, cv2.RANSAC, 20.0)[0]
    
    # draw_matches(possiblyCorrectMatches, source_img, source_keypoints, target_img, target_keypoints)
    # numMatches = len(possiblyCorrectMatches)
    # for i in range(numMatches):
    #     xy = H @ np.array([[sourcepts[i,0]], [sourcepts[i,1]], [1]])
    #     x = xy[0]/xy[2]
    #     y = xy[1]/xy[2]
    
    return H
    
def find_board(target_file, source_file):
    source_img = cv2.imread(source_file, 1)
    if source_img is None:
        raise Exception(f"'{source_file}' could not opened")
    
    target_img = cv2.imread(target_file, 1)
    if target_img is None:
        raise Exception(f"'{target_file}' could not opened")

    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    H = find_homography_between_images(source_img, target_img)
        
    warped = cv2.warpPerspective(source_img, H, (target_img.shape[1], target_img.shape[0]))
    return warped, target_img

if __name__ == "__main__":
    target_file="assets/0.0 Cropped/11.png"
    source_file="assets/2.0 Red-Red/PXL_20220209_150018426.jpg"
    # for source_file in getImagesInDir("assets/2.0 Red-Red"):
        
    board = find_board(target_file, source_file)
    
    # annotated = annotate_fixed_city_points(np.copy(board), "assets/0.0 Cropped/cities11.csv")
    # plt.imshow(annotated)
    plt.imshow(board)
    plt.show()
    
