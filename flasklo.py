from flask import Flask, render_template, request, redirect, url_for
from imutils import face_utils
import dlib
import cv2
from pygame import mixer
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from twilio.rest import Client
import math
import requests

app = Flask(__name__)

# Function to send email with image attachment and live location link
def send_email(receiver_email, image, location_link):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "pythoncoder028@gmail.com"
    password = "doctxgetwmzyqkbu"  # App password generated from Gmail

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Drowsiness Detected!"

    body = """\
    Dear User,

    Drowsiness has been detected. Please be alert.

    Regards,
    Your Drowsiness Detection System
    
    
    """

    message.attach(MIMEText(body, "plain"))

    # Attaching the image
    img_attachment = MIMEImage(image)
    img_attachment.add_header('Content-Disposition', 'attachment', filename='drowsiness_image.jpg')
    message.attach(img_attachment)

    # Attaching the live location link
    message.attach(MIMEText(location_link, "html"))

    # Attaching the audio file
    with open("alarm.wav", "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name="alarm.wav")

    part['Content-Disposition'] = f'attachment; filename="alarm.wav"'
    message.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Function to send WhatsApp message
def send_whatsapp(receiver_phone_number, message):
    account_sid = 'AC4e946bb7a304070016fabb96def57b6c'
    auth_token = '504616d05f2ab3ea488717299139f1af'
    twilio_phone_number = '+14155238886'

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message,
        from_='whatsapp:' + twilio_phone_number,
        to='whatsapp:' + receiver_phone_number
    )

# Function to calculate Euclidean distance between two points
def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to get live location using IP address
def get_live_location():
    try:
        ip_data = requests.get("https://ipinfo.io/json").json()
        latitude, longitude = ip_data["loc"].split(",")
        location_link = f'https://maps.google.com/?q={latitude},{longitude}'
        return location_link
    except Exception as e:
        print("Error occurred while fetching live location:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        receiver_email = request.form['email']
        receiver_whatsapp_number = request.form['whatsapp_number']

        # Start the drowsiness detection process
        thres = 6
        print("Initializing mixer...")
        try:
            mixer.init()
            print("Mixer initialized successfully.")
        except Exception as e:
            print("Error initializing mixer:", e)

        mixer.init()
        sound = mixer.Sound('alarm.wav')
        dlist = []

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return redirect(url_for('index'))

        while True:
            ret, image = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                le_38 = shape[37]
                le_39 = shape[38]
                le_41 = shape[40]
                le_42 = shape[41]

                re_44 = shape[43]
                re_45 = shape[44]
                re_47 = shape[46]
                re_48 = shape[47]

                dlist.append((dist(le_38, le_42) + dist(le_39, le_41) + dist(re_44, re_48) + dist(re_45, re_47)) / 4 < thres)
                if len(dlist) > 10:
                    dlist.pop(0)

                if sum(dlist) >= 4:
                    try:
                        sound.play()
                        location_link = get_live_location()
                        if location_link:
                            send_email(receiver_email, cv2.imencode('.jpg', image)[1].tobytes(), location_link)
                            print("Email sent with live location and sound played.")

                            whatsapp_message = f"""\
Dear User,

We've detected signs of drowsiness. Please ensure your safety by staying alert.

Your Drowsiness Detection System

Live Location: {location_link}"""

                            send_whatsapp(receiver_whatsapp_number, whatsapp_message)
                            print("WhatsApp alert with live location has been sent")
                            
                            # Break the loop and exit after sending email and WhatsApp message
                            break
                        else:
                            print("Error: Failed to get live location.")
                    except Exception as e:
                        print("Error occurred:", e)
                else:
                    try:
                        sound.stop()
                    except Exception as e:
                        print("Error occurred:", e)

            cv2.imshow("Output", image)

            key = cv2.waitKey(1)
            if key == 27:  # Esc key
                break

        cap.release()
        cv2.destroyAllWindows()

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
