import os 
import cv2


# To read the image --> file
sample = cv2.imread("./Kaggle/images/101__M_Left_little_finger.bmp")

# How good or close is the current match vs the best match, and if so, replace the best match
best_score = 0

filename = None

# This will be the best image
image = None

# We want to have key points of the "Sample" and also "original" image
# so we can plot the connections between the individual key points.
kp1, kp2, mp = None, None, None

counter = 0 

# This for loop will go through the list or directory up to 1000 images
for file in [file for file in os.listdir("./Kaggle/images")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1
    
    fingerprint_image = cv2.imread("./Kaggle/images/" + file)
    sift = cv2.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                    {}).knnMatch(descriptors_1, descriptors_2, k=2)
    
    match_points = []
    
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
            
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
        
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points
        
print("BEST MATCH: " + filename)
print("SCORE " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
