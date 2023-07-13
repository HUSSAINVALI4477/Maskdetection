import cv2
import numpy as np

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained mask detection model from OpenCV
mask_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load the input image
image = cv2.imread('test_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop over each detected face
for (x, y, w, h) in faces:
    # Extract the face region of interest (ROI)
    face = image[y:y+h, x:x+w]

    # Preprocess the face for mask detection
    blob = cv2.dnn.blobFromImage(face, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the face blob through the mask detection network
    mask_net.setInput(blob)
    detections = mask_net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Determine the class label and color for the bounding box
            label = "Mask" if detections[0, 0, i, 1] == 0 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display the label and bounding box rectangle on the output image
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
