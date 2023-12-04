import face_recognition
import cv2
import numpy as np

# Load a sample picture and learn how to recognize it.
adi_image = face_recognition.load_image_file("imgs_database/adi.jpg")
adi_fc = face_recognition.face_encodings(adi_image)[0]

#face_encoding=fc

# Load  2nd sample picture and learn how to recognize it.
arjun_image = face_recognition.load_image_file("imgs_database/arjun.jpg")
arjun_fc = face_recognition.face_encodings(arjun_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    adi_fc,arjun_fc
]

known_face_names = [
    "Adithya","Arjun"
]

names_copy=known_face_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

confidence_threshold = 0.5

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
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=confidence_threshold)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

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
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a circle around the face
        center = ((left + right) // 2, (top + bottom) // 2)
        radius = (right - left) // 2
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left+40, bottom - 35), (right-40, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size, _ = cv2.getTextSize(name, font, 1.0, 1)
        text_x = left + (right - left - text_size[0]) // 2  # Calculate x-coordinate for center alignment
        cv2.putText(frame, name, (text_x, bottom - 6), font, 1.0, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()