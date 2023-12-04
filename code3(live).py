import face_recognition
import cv2
import numpy as np

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
confidence_threshold = 0.5

# Set the liveness detection threshold (adjust as needed)
movement_threshold = 10

# Initialize variables for liveness detection
last_known_face_location = None
movement_count = 0

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=confidence_threshold)
            name = "Unknown"
    
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
                # Liveness detection: Check for facial movement
                if last_known_face_location is not None:
                  movement = np.linalg.norm(np.array([top, right, bottom, left]) - last_known_face_location)
                  if movement > movement_threshold:
                     print(f'{name} is likely a live person.')
                  else:
                     print(f'{name} may not be a live person (little movement).')
                     
                # Update the last known face location
                last_known_face_location = np.array([top, right, bottom, left])

            face_names.append(name)
            if name in known_face_names:
                if name in names_copy:
                    names_copy.remove(name)
                    print('     ' + name + ' : detected with confidence {:.2f}'.format(1 - face_distances[best_match_index]))

    process_this_frame = not process_this_frame
    

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
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