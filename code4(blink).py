import face_recognition
import cv2
import numpy as np

# Load your known faces and encodings
# ... (previous code)
# Load a sample picture and learn how to recognize it.
adi_image = face_recognition.load_image_file("imgs_database/adi.jpg")
adi_fc = face_recognition.face_encodings(adi_image)[0]


# Load  2nd sample picture and learn how to recognize it.
arjun_image = face_recognition.load_image_file("imgs_database/arjun.jpg")
arjun_fc = face_recognition.face_encodings(arjun_image)[0]

sars_image = face_recognition.load_image_file("imgs_database/sarsmech.jpg")
sars_fc = face_recognition.face_encodings(sars_image)[0]

karti_image = face_recognition.load_image_file("imgs_database/karthi.jpg")
karti_fc = face_recognition.face_encodings(karti_image)[0]

gk_image = face_recognition.load_image_file("imgs_database/gkmech.jpg")
gk_fc = face_recognition.face_encodings(gk_image)[0]

pranav_image = face_recognition.load_image_file("imgs_database/pranav.jpg")
pranav_fc = face_recognition.face_encodings(pranav_image)[0]

ismail_image = face_recognition.load_image_file("imgs_database/ismail.jpg")
ismail_fc = face_recognition.face_encodings(ismail_image)[0]

sharn_image = face_recognition.load_image_file("imgs_database/sharn.jpg")
sharn_fc = face_recognition.face_encodings(sharn_image)[0]

hari_image = face_recognition.load_image_file("imgs_database/hari.jpg")
hari_fc = face_recognition.face_encodings(hari_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    adi_fc,arjun_fc,sars_fc,karti_fc,gk_fc,pranav_fc,ismail_fc,sharn_fc,hari_fc
]

known_face_names = [
    "Adithya","Arjun","Sheik Masthan","Karthick","Dr.G.Kanagaraj","Pranav","Ismail","Sharanya","Harikrishna"
]

names_copy=known_face_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# Set the face recognition confidence threshold
confidence_threshold = 0.6

# Initialize variables for liveness detection
last_known_face_encoding = None
challenge_threshold = 20
challenge_issued = False

# Open a video capture object
video_capture = cv2.VideoCapture(0)

name='unknown'

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Check for movement in facial landmarks
        if last_known_face_encoding is not None:
            movement = np.linalg.norm(np.array(face_encoding) - np.array(last_known_face_encoding))
            if movement > challenge_threshold and not challenge_issued:
                # Face is moving (considered live), issue a challenge
                print("Blink your eyes to complete the challenge.")
                challenge_issued = True
            elif movement <= challenge_threshold and challenge_issued:
                # Face is not moving, reset the challenge
                challenge_issued = False

            if challenge_issued:
                # Additional challenge-response logic
                # For example, wait for a blink to confirm liveness
                # You can use a dedicated blink detection library or implement custom logic
                # Here, we use a simple time-based approach for demonstration purposes
                # You may need to install the dlib library for this example: pip install dlib
                import dlib
                predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to facial landmark predictor
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(predictor_path)
                landmarks = predictor(rgb_small_frame, dlib.rectangle(left, top, right, bottom))
                left_eye = landmarks.part(43).y - landmarks.part(40).y
                right_eye = landmarks.part(47).y - landmarks.part(44).y
                if left_eye > 5 and right_eye > 5:
                    # Blink detected, challenge successfully completed
                    print("Challenge completed. Face is live.")
                    # Perform face recognition if needed
            else:
                # Normal face recognition logic
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=confidence_threshold)
                name = "Unknown"
                if any(matches):
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]
                print(f"{name} is a live person.")
        else:
            # First frame, initialize last_known_face_encoding
            last_known_face_encoding = face_encoding

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # ... (previous code)
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
