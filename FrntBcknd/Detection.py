import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QDialog, QLineEdit, \
    QMessageBox, QSpacerItem, QSizePolicy, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import boto3
import os
from datetime import datetime

from PyQt5.QtCore import Qt  # Import Qt module


class CameraFrame(QWidget):
    def __init__(self, email):
        super().__init__()
        # some parts of the codes were remove for security purposes - dev Mark

        self.yolo_model = YOLO('best 1.pt')  # Replace with your YOLO model path
        self.video_path = 'kicking_ 1.mp4'  # Replace with your video file path
        self.camera_label = QLabel()
        self.toggle_button = QPushButton('Toggle Camera')
        self.toggle_button.setCheckable(True)
        self.camera_on = False
        self.image_counter = 0  # Counter to maintain image filenames
        # Set the email address
        self.email = email

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.toggle_button)
        self.setLayout(layout)

        self.logout_button = QPushButton('Logout')
        layout.addWidget(self.logout_button)

        self.logout_button.clicked.connect(self.logout)

        self.toggle_button.clicked.connect(self.toggle_camera)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)

        self.labels = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.class_thresholds = {
            "peeking": 0.5,
            "lockpicking": 0.5,
            "kicking": 0.5,
            "shoulderslam": 0.5
        }

    def logout(self):
        reply = QMessageBox.question(self, 'Logout', 'Are you sure you want to logout? This will close the program.',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Display a message before closing
            message_box = QMessageBox(QMessageBox.Information, "Closing", "Closing... Click OK")
            message_box.exec_()
            QApplication.quit()
        else:
            # If 'No' is clicked, return to the camera frame (do nothing here as the method will just return)
            pass

    def load_model(self):
        model = YOLO("best 1.pt")  # Load the yolov8m-seg-Custom-Dataset.pt model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            conf = boxes.conf[0]
            xyxy = boxes.xyxy[0]

            class_name = self.CLASS_NAMES_DICT[class_id]
            if class_name in self.class_thresholds and conf >= self.class_thresholds[class_name]:
                xyxys = result.boxes.xyxy.cpu().numpy()

                for box in xyxys:
                    x, y, x_max, y_max = map(int, box[:4])  
                    label = f"{class_name}: {conf:.2f}"
                    cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save the frame where the bounding box appears as an image
                    self.capture_frame(frame)

                    # Upload the captured image to S3 and send its URL to DynamoDB
                    image_url = self.upload_image_to_s3()
                    self.send_to_dynamodb(class_name, "Suspicious Action Detected.", image_url)

        return frame

    def capture_frame(self, frame):
        # Create a folder based on the email address if it doesn't exist
        email_folder = self.email
        os.makedirs(email_folder, exist_ok=True)

        # Save the frame as an image locally
        image_path = f"{email_folder}/frame_{self.image_counter}.jpg"
        cv2.imwrite(image_path, frame)
        self.image_counter += 1  # Increment image counter

    def upload_image_to_s3(self):
        # Fetch the latest saved frame
        image_filename = f"{self.email}/frame_{self.image_counter - 1}.jpg"

        # Upload the captured image to AWS S3
        with open(image_filename, "rb") as file:
            self.s3.upload_fileobj(file, self.bucket_name, image_filename)

        # Get the S3 image URL
        image_url = f"https://{self.bucket_name}.s3.amazonaws.com/{image_filename}"
        return image_url

    def send_to_dynamodb(self, label, title, image_url):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get the item from the DynamoDB table
        response = self.table.get_item(Key={'email': self.email})
        item = response.get('Item', {})

        # Check if the attributes are present and if they are lists
        messages = item.get('messages', [])
        if not isinstance(messages, list):
            messages = []
        title_list = item.get('title', [])
        if not isinstance(title_list, list):
            title_list = []
        images = item.get('images', [])
        if not isinstance(images, list):
            images = []
        times = item.get('times', [])
        if not isinstance(times, list):
            times = []

        # Append new values to the respective lists
        messages.append(label)
        title_list.append("Suspicious Action Detected.")
        images.append(image_url)
        times.append(current_time)

        # Update the DynamoDB table with the new values
        response = self.table.update_item(
            Key={
                'email': self.email
            },
            UpdateExpression="SET messages = :m, title = :t, images = :i, times = :time",
            ExpressionAttributeValues={
                ':m': messages,
                ':t': title_list,
                ':i': images,
                ':time': times
            },
            ReturnValues="UPDATED_NEW"
        )
        print("Label, Title, Image URL, and Time sent to DynamoDB:", label, title, image_url, current_time)

    def toggle_camera(self):
        if not self.camera_on:
            self.camera_on = True
            self.toggle_button.setText('Stop Camera')
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(10)  # Adjust the time interval (milliseconds) for smoother playback
        else:
            self.camera_on = False
            self.toggle_button.setText('Start Camera')
            self.cap.release()
            self.timer.stop()

    def update_camera(self):
        ret, frame = self.cap.read()

        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                results = self.predict(frame)
                frame = self.plot_bboxes(results, frame)

                
            except Exception as e:
                print(f"Error during inference: {e}")
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)
            self.camera_label.setScaledContents(True)
                
        else:
            print("Error reading frame")
            self.toggle_camera()


class RegisterWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Register")
        self.layout = QVBoxLayout()
        self.setFixedSize(500, 500)  # Set the fixed size of the MainWindow


        self.email_label = QLabel("Email:")
        self.email_entry = QLineEdit()
        self.layout.addWidget(self.email_label)
        self.layout.addWidget(self.email_entry)

        self.password_label = QLabel("Password:")
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_entry)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register_account)
        self.layout.addWidget(self.register_button)

        self.setLayout(self.layout)

    def register_account(self):
        signUpEmail = self.email_entry.text()
        signUpPassword = self.password_entry.text()

        if signUpEmail == '' or signUpPassword == '':
            QMessageBox.critical(self, 'Error', 'All fields need to be filled')
        else:
            
            table = dynamodb.Table('LOID_Client')

            response = table.get_item(Key={'email': signUpEmail})
            if 'Item' in response:
                QMessageBox.critical(self, "Registration", "User already registered. Please log in.")
            else:
                table.put_item(Item={'email': signUpEmail, 'password': signUpPassword})
                QMessageBox.information(self, "Registration", "You are now registered. Please proceed to Login.")
                self.accept()  # Close the registration window and proceed to login


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LOID Login")
        self.setFixedSize(500, 500)  # Set the fixed size of the MainWindow

        self.setStyleSheet(
            "QDialog {"
            "background-color: #F1EFEF;"
            "}"
        )

        layout = QVBoxLayout()

        #email area
        self.email_label = QLabel("Email:")
        self.email_entry = QLineEdit()
        layout.addWidget(self.email_label)
        layout.addWidget(self.email_entry)

        #password area
        self.password_label = QLabel("Password:")
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_entry)


        #login button area
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login_account)
        layout.addWidget(self.login_button)

        self.setLayout(layout)



    def login_account(self):
        loginEmail = self.email_entry.text()
        loginPassword = self.password_entry.text()

        if loginEmail == '' or loginPassword == '':
            QMessageBox.critical(self, 'Error', 'All fields need to be filled')
            table = dynamodb.Table('LOID_Client')

            response = table.get_item(Key={'email': loginEmail})
            if 'Item' not in response:
                QMessageBox.critical(self, "Login", "User not found. Please register first.")
            else:
                user_data = response.get('Item', {})
                if user_data.get('password') == loginPassword:
                    QMessageBox.information(self, "Login", "Login Successful!")
                    self.accept()  # Close the login window and proceed to main window
                else:
                    QMessageBox.critical(self, "Login", "Incorrect password. Enter your details again.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LOID Detection")
        self.setGeometry(100, 100, 640, 480)
        self.setFixedSize(800, 800)  # Set the fixed size of the MainWindow

        # Create buttons for Register and Login
        self.register_button = QPushButton("REGISTER")
        self.login_button = QPushButton("LOGIN")

        button_font = QFont("Poppins Medium", 16)
        self.register_button.setFont(button_font)
        self.login_button.setFont(button_font)

        # Connect buttons to their respective windows
        self.register_button.clicked.connect(self.open_register_window)
        self.login_button.clicked.connect(self.open_login_window)

        # Set button size
        button_width = 500
        button_height = 100
        self.register_button.setFixedSize(button_width, button_height)
        self.login_button.setFixedSize(button_width, button_height)

        # Set button color using style sheets
        button_stylesheet = (
            "QPushButton {"
            "background-color: #67729D;"  # Change color here (e.g., green color)
            "color: white;"
            "border-radius: 10px;"
            "}"
            "QPushButton:hover {"
            "background-color: #9D76C1;"  # Change hover color here
            "}"
        )
        self.register_button.setStyleSheet(button_stylesheet)
        self.login_button.setStyleSheet(button_stylesheet)

        # Set background image for the main window
        self.setStyleSheet(
            "QMainWindow {"
            "background-image: url('loid.jpg');"
            "}"
        )

        # Set layout for the main window
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        top_spacer = QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(top_spacer)

        # Add QLabel to display the image at the top
        image_label = QLabel(self)
        pixmap = QPixmap('ic_notification - Copy.png')  # Load your image here
        pixmap = pixmap.scaledToWidth(400)  # Adjust the width as needed
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # Add QLabel to layout
        layout.addWidget(image_label)

        # Add space between image and Register button
        layout.addSpacing(120)

        layout.addWidget(self.register_button)
        layout.addSpacing(30)  # Add space between buttons (adjust as needed)
        layout.addWidget(self.login_button)

        bottom_spacer = QSpacerItem(40, 80, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(bottom_spacer)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_register_window(self):
        register_window = RegisterWindow()
        register_window.exec_()

    def open_login_window(self):
        login_window = LoginWindow()
        if login_window.exec_() == QDialog.Accepted:  # Check for successful login
            email = login_window.email_entry.text()

            # Create and display the CameraFrame widget after successful login
            self.camera_frame = CameraFrame(email)
            self.setCentralWidget(self.camera_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
