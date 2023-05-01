import os 
import cv2

# Read the sample image file
sample = cv2.imread("./Kaggle/images/101__M_Left_little_finger.bmp")

# Set initial values for best_score and filename
best_score = 0
filename = None

# Set initial values for key points
kp1, kp2, mp = None, None, None

# Set counter to 0
counter = 0

# Loop through each file in the directory up to 1000 images
for file in [file for file in os.listdir("./Kaggle/images")][:1000]:
    
    # Check if counter is a multiple of 10 and print the current file being processed
    if counter % 10 == 0:
        print(counter)
        print(file)
        
    # Increment counter by 1
    counter += 1
    
    # Read the fingerprint image file
    fingerprint_image = cv2.imread("./Kaggle/images/" + file)
    
    # Create a SIFT object
    sift = cv2.SIFT_create()
    
    # Detect and compute the key points and descriptors of the sample and fingerprint images
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    
    # Match the descriptors using the FLANN-based matcher
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    
    # Filter the matches using the ratio test and store the good matches
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
            
    # Determine the number of keypoints to calculate the score
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
        
    # Calculate the score
    score = len(match_points) / keypoints * 100
    
    # Update the best_score and filename if the current score is higher than the previous best_score
    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points
        
# Print the best match and its score
print("BEST MATCH: " + filename)
print("SCORE " + str(best_score))

# Draw the matches and display the result
result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
