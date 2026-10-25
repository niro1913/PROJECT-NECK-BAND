import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# ============================================================
# TTS
# ============================================================

from gtts import gTTS
import tempfile
import os
import threading
import queue
from playsound import playsound


# ============================================================
# 1. CLASS NAMES
# ============================================================

class_names = [
    "10",
    "100",
    "20",
    "200",
    "2000",
    "50",
    "500",
    "Background"
]


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "currency_model.pth",
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 4. OPPO A78 IP WEBCAM
# ============================================================

url = "http://100.124.231.64:8080/video"

cap = cv2.VideoCapture(
    url,
    cv2.CAP_FFMPEG
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print("Camera connection failed!")
    print("Check your IP Webcam IP address.")
    exit()


print("==============================================")
print("     INDIAN CURRENCY RECOGNITION + TTS")
print("==============================================")
print("Camera connected successfully!")
print("Show an Indian currency note.")
print("The detected value will be spoken.")
print("Press Q to stop.")
print("==============================================")


# ============================================================
# 5. TTS QUEUE
# ============================================================

tts_queue = queue.Queue(
    maxsize=1
)

tts_running = True


# ============================================================
# 6. TTS FUNCTION
# ============================================================

def speak_currency(currency):

    try:

        # Convert:
        # "500" -> "500 rupees"

        speech_text = (
            f"{currency} rupees"
        )

        print(
            f"Speaking: {speech_text}"
        )

        # Generate speech
        tts = gTTS(
            text=speech_text,
            lang="en",
            slow=False
        )

        # Temporary MP3 file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as fp:

            temp_file = fp.name


        # Save speech
        tts.save(
            temp_file
        )


        # Play speech
        playsound(
            temp_file
        )


        # Delete temporary file
        try:

            os.remove(
                temp_file
            )

        except:

            pass


    except Exception as e:

        print(
            "TTS error:",
            e
        )


# ============================================================
# 7. TTS WORKER THREAD
# ============================================================

def tts_worker():

    while tts_running:

        try:

            currency = tts_queue.get(
                timeout=0.2
            )

        except queue.Empty:

            continue


        if currency is None:

            continue


        speak_currency(
            currency
        )


tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()


# ============================================================
# 8. VARIABLES
# ============================================================

frame_count = 0

predicted_class = "Background"

confidence_value = 0

last_spoken_currency = None

# Prevent immediate repeated speech
last_detection_time = 0

# Seconds before the same currency can be spoken again
SPEAK_COOLDOWN = 3


# ============================================================
# 9. LIVE DETECTION
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "Camera frame failed!"
        )

        break


    frame_count += 1


    # ========================================================
    # RUN AI EVERY 10 FRAMES
    # ========================================================

    if frame_count % 10 == 0:


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # OpenCV -> PIL
        # ----------------------------------------------------

        image = Image.fromarray(
            rgb_frame
        )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        input_image = transform(
            image
        ).unsqueeze(0)


        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_image
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_class = class_names[
            prediction.item()
        ]


        confidence_value = (
            confidence.item()
            * 100
        )


        # ====================================================
        # SPEAK ONLY WHEN CONFIDENT
        # ====================================================

        current_time = cv2.getTickCount() / cv2.getTickFrequency()


        # Don't speak Background
        if predicted_class != "Background":


            # Only speak if confidence is good
            if confidence_value >= 70:


                # Speak when:
                # 1. Currency changed
                # OR
                # 2. Cooldown has passed

                currency_changed = (
                    predicted_class
                    !=
                    last_spoken_currency
                )


                cooldown_finished = (
                    current_time
                    -
                    last_detection_time
                    >=
                    SPEAK_COOLDOWN
                )


                if (
                    currency_changed
                    or
                    cooldown_finished
                ):


                    # Don't allow multiple queued
                    # messages

                    try:

                        while True:

                            tts_queue.get_nowait()

                    except queue.Empty:

                        pass


                    try:

                        tts_queue.put_nowait(
                            predicted_class
                        )

                        last_spoken_currency = (
                            predicted_class
                        )

                        last_detection_time = (
                            current_time
                        )

                    except queue.Full:

                        pass


        else:

            # When background appears,
            # allow the next currency to speak

            if confidence_value >= 60:

                last_spoken_currency = None


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if predicted_class == "Background":

        text = (
            "No currency detected"
        )

    else:

        text = (
            f"Currency Detected: "
            f"Rs.{predicted_class}"
        )


    # ========================================================
    # DISPLAY CURRENCY
    # ========================================================

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: {confidence_value:.2f}%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # TTS STATUS
    # ========================================================

    if predicted_class != "Background":

        tts_display = (
            "Speaking currency..."
        )

    else:

        tts_display = (
            "Show currency note"
        )


    cv2.putText(
        frame,
        tts_display,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Indian Currency Recognition",
        frame
    )


    # ========================================================
    # Q = QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

tts_running = False

try:

    tts_queue.put_nowait(
        None
    )

except:

    pass


cap.release()

cv2.destroyAllWindows()

print(
    "Program stopped."
)