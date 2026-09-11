import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

# Define the classes for your gestures
classes = {
    0: 'Fist',
    1: 'Five',
    2: 'None',
    3: 'Okay',
    4: 'Peace',
    5: 'Rad',
    6: 'Straight',
}
image_size=[150,150]


roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


def build_model():
    cur_path = os.getcwd()
    train_dir = os.path.join(cur_path, "data", 'train')
    validation_dir = os.path.join(cur_path, "data", 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=4,
        class_mode='sparse')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=4,
        class_mode='sparse')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(7, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history=model.fit(
        train_generator,
        steps_per_epoch=25,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=5,
        verbose=2)

    # Plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    model.save("hand_gesture_model.h5")

def preprocess_hand_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_background = np.zeros_like(image)
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    hand_image = cv2.resize(black_background, image_size)
    hand_image = cv2.flip(hand_image, 1)

    # Invert the colors
    hand_image = cv2.bitwise_not(hand_image)

    # Normalize the image
    hand_image = hand_image / 255.0
    cv2.imshow("", hand_image)
    # cv2.waitKey(0)

    # Reshape the image
    hand_image = np.reshape(hand_image, (1, 150, 150, 3))
    return hand_image

def classify_image(image):
    model = load_model('hand_gesture_model.h5')
    image = preprocess_hand_image(image)
    # image = cv2.resize(image, image_size)
    # image = np.reshape(image, (1, 150, 150, 3))
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    print(classes[predicted_class])
    return classes[predicted_class]

def classify_hand():
    # model = load_model('hand_gesture_model.h5')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        # cv2.imshow("",roi)

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)
        detected_class = classify_image(roi)

        cv2.putText(frame, detected_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # build_model()
    path = "straight.jpg"
    image = cv2.imread(path)
    # build_model()
    # classify_image(image)
    classify_hand()
