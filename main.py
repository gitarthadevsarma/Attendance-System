import cv2
import face_recognition
import csv
import datetime

# Load the known faces and their names
salman_image = face_recognition.load_image_file("salman.jpg")
salman_face_encoding = face_recognition.face_encodings(salman_image)[0]

akshay_image = face_recognition.load_image_file("akshay.jpg")
akshay_face_encoding = face_recognition.face_encodings(akshay_image)[0]

hritik_image = face_recognition.load_image_file("hritik.jpg")
hritik_face_encoding = face_recognition.face_encodings(hritik_image)[0]

katrina_image = face_recognition.load_image_file("katrina.jpg")
katrina_face_encoding = face_recognition.face_encodings(katrina_image)[0]

kriti_image = face_recognition.load_image_file("kriti.jpg")
kriti_face_encoding = face_recognition.face_encodings(kriti_image)[0]

tapsi_image = face_recognition.load_image_file("tapsi.jpg")
tapsi_face_encoding = face_recognition.face_encodings(tapsi_image)[0]

sarukh_image = face_recognition.load_image_file("sarukh.jpg")
sarukh_face_encoding = face_recognition.face_encodings(sarukh_image)[0]

known_face_encodings = [salman_face_encoding, akshay_face_encoding, hritik_face_encoding, katrina_face_encoding, kriti_face_encoding, tapsi_face_encoding, sarukh_face_encoding]
known_face_names = ["Salman Khan", "akshay Kumar", "hritik roshan", "katrina kaif", "kriti sanon", "tapsi pannu", "Sarukh Khan"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Create a list to keep track of attendance
attendance = []

# Open a video capture stream
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Capture a single frame from the video stream
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Record the attendance
            attendance.append((name, datetime.datetime.now()))

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name of the person
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
video_capture.release()
cv2.destroyAllWindows()

# Save the attendance data to a CSV file
with open('attendance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Time'])
    for entry in attendance:
        writer.writerow(entry)
