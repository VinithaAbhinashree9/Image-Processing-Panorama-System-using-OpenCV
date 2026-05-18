# Import libraries
import cv2
import numpy as np


image = cv2.imread(r"C:\Users\MR\Desktop\module08\image_processing_project_q2\image1.jpg")

if image is None:
    print("Image not found!")
    exit()

cv2.imshow("Original Image", image)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)


# Convert image into black and white
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Threshold Image", threshold)


# Gaussian Blur
blur = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow("Blurred Image", blur)


edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Edge Detection", edges)


# Detect corners using Harris Corner Detection
gray_float = np.float32(gray)

corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)

# Highlight detected corners
image_copy = image.copy()
image_copy[corners > 0.01 * corners.max()] = [0, 0, 255]

cv2.imshow("Feature Detection", image_copy)


# Read second image
image2 = cv2.imread(r"C:\Users\MR\Desktop\module08\image_processing_project_q2\image2.jpg")

if image2 is not None:

    # Create stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch images
    status, panorama = stitcher.stitch([image, image2])

    if status == cv2.STITCHER_OK:
        cv2.imshow("Panorama Image", panorama)
        cv2.imwrite("panorama.jpg", panorama)
        print("Panorama created successfully!")
    else:
        print("Panorama stitching failed!")

else:
    print("Second image not found!")



cv2.imwrite("threshold.jpg", threshold)
cv2.imwrite("blurred.jpg", blur)
cv2.imwrite("edges.jpg", edges)

print("Processed images saved successfully!")



cv2.waitKey(0)
cv2.destroyAllWindows()