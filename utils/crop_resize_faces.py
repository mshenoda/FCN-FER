import os
import cv2
import argparse

def crop_resize_faces(input_dir, output_dir):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Loop through the input directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Crop and resize the faces
                for (x, y, w, h) in faces:
                    # Add padding to the face
                    padding_top = int(0.2 * h)
                    padding_bottom = int(0.1 * h)
                    y = max(0, y - padding_top)
                    h += padding_top + padding_bottom

                    # Crop the face
                    face = image[y:y+h, x:x+w]

                    # Resize the face to 66x88
                    face = cv2.resize(face, (48, 48))

                    # Save the cropped and resized face with JPG quality of 97
                    output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path[:-4] + '_roi.jpg', face, [int(cv2.IMWRITE_JPEG_QUALITY), 97])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect faces in images and save cropped faces to output directory.')
    parser.add_argument('input_dir', type=str, help='path to input directory')
    parser.add_argument('output_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    crop_resize_faces(args.input_dir, args.output_dir)
