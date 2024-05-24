import os
import cv2
from config import VIDEO_FILES_DIR, FRAMES_FILES_DIR, VIDEO_DATASET_DIR

# To keep track of frames with no face or multiple faces
no_face = []
multiple_faces = []

def prepare_all_videos(filenames, paths, output_path, resolution, skip=1):
    # Create output directory
    resolution_name = f'_{resolution[0]}x{resolution[1]}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for count, video in enumerate(zip(filenames, paths)):
        # Gather all its frames
        save_frames(filename=video[0], input_path=video[1], output_path=output_path, resolution=resolution,skip=skip)
        print(f"Processed videos {count+1}/{len(paths)}")
    return

def save_frames(filename, input_path, output_path, resolution, skip):
    # Initialize video reader
    cap = cv2.VideoCapture(input_path + '.mp4')
    haar_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')
    frames_count = 0

    # Retrieve frame count from cap
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Retrieve video length from cap
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

    # Compute frame for each second
    frames_per_second = total_frames // video_length

    try:
        # Loop through all frames
        while cap.read():
            # Capture frame
            ret, frame = cap.read()

            # TODO: uncomment, for testing purposes 👁️👄👁️
            # Skip the first and last second of the video
            # if frames_count >= frames_per_second and frames_count <= (total_frames - frames_per_second):
            if not ret:
                break

            # Detect faces
            faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                
            if len(faces) == 0: # No face detected
                faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9) # Try again with different parameters
                if len(faces) == 0: # Still no face detected
                    # Still no face, save frame name in a list for manual inspection and continue
                    print(f"No face detected in {filename}")
                    no_face.append(filename)
                    # Save list on disk
                    with open(VIDEO_DATASET_DIR + '/TOREMOVE_no_face.txt', 'w') as f:
                        for item in no_face:
                            f.write("%s\n" % item)

            if len(faces) > 1: # More than one face detected
                # More than one face detected, save frame name in a list for manual inspection and continue
                print(f"More than one face detected in {filename}")
                multiple_faces.append(filename)
                # Save list on disk
                with open(VIDEO_DATASET_DIR + '/TOREMOVE_multiple_faces.txt', 'w') as f:
                    for item in multiple_faces:
                        f.write("%s\n" % item)

            for (x, y, w, h) in faces: # Save face
                face = frame[y:y + h, x:x + w] # Crop face

            # face = black_background(face) # Remove white background
            
            face = cv2.resize(face, resolution) # Resize face

            cv2.imwrite(output_path + f'/{filename}_{frames_count}' + '.png', face) # Save face
            frames_count += 1
    finally:
        cap.release()
    return

def black_background(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # background from white to black
    ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    face[thresh == 255] = 0

    return face

if __name__ == "__main__":
    output_path = FRAMES_FILES_DIR # + '_black_background'
    resolution = (224, 224)

    filenames = []
    feats = []
    labels = []
    paths = []

    for (dirpath, dirnames, fn) in os.walk(VIDEO_FILES_DIR):
        for name in fn:
            filename = name.split('.')[0]
            feat = filename.split('-')[2:]
            label = feat[0]
            filenames.append(filename)
            feats.append(feat)
            labels.append(label)
            paths.append(dirpath + '/' + filename)
            
    prepare_all_videos(filenames, paths, output_path, resolution, skip=3)
