import cv2
import pytesseract
import threading
import time
import re
import numpy as np
import tempfile
import os
import queue

from gtts import gTTS
from playsound import playsound


# ============================================================
# TESSERACT
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# IP WEBCAM
# ============================================================

URL = "http://100.124.231.64:8080/video"

cap = cv2.VideoCapture(
    URL,
    cv2.CAP_FFMPEG
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print("Camera connection failed!")
    exit()


print("==============================================")
print("        SMART REAL-TIME OCR + TTS")
print("==============================================")
print("Camera connected successfully!")
print("Show text to the camera.")
print("Click SPEAK to hear the text.")
print("Click CLEAR to clear the OCR.")
print("Press Q to quit.")
print("==============================================")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

latest_frame = None

frame_lock = threading.Lock()

running = True


# ============================================================
# OCR VARIABLES
# ============================================================

ocr_text = "Looking for text..."

ocr_confidence = 0.0

ocr_lock = threading.Lock()


# ============================================================
# TTS VARIABLES
# ============================================================

tts_queue = queue.Queue(maxsize=1)

tts_status = "Ready"

tts_status_lock = threading.Lock()

last_spoken_text = ""


# ============================================================
# BUTTON VARIABLES
# ============================================================

button_clicked = None

button_lock = threading.Lock()


button_positions = {
    "speak": (0, 0, 0, 0),
    "clear": (0, 0, 0, 0)
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{2,}",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# CHECK VALID TEXT
# ============================================================

def is_valid_text(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 2:
        return False

    letters = sum(
        c.isalpha()
        for c in text
    )

    digits = sum(
        c.isdigit()
        for c in text
    )

    symbols = sum(
        not c.isalnum()
        and not c.isspace()
        for c in text
    )

    total = len(text)

    if total == 0:
        return False

    symbol_ratio = symbols / total

    # Reject mostly-symbol garbage
    if symbol_ratio > 0.45:
        return False

    if letters >= 2:
        return True

    if digits >= 2:
        return True

    return False


# ============================================================
# ROTATE CAMERA
# ============================================================

def rotate_camera(frame):

    return cv2.rotate(
        frame,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale for OCR
    gray = cv2.resize(
        gray,
        None,
        fx=1.7,
        fy=1.7,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # Mild noise reduction
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


# ============================================================
# OCR
# ============================================================

def perform_ocr(frame):

    # Rotate for OCR
    frame = rotate_camera(
        frame
    )

    # Preprocess
    gray = preprocess_image(
        frame
    )

    # ========================================================
    # OCR
    # ========================================================

    data = pytesseract.image_to_data(
        gray,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )


    # ========================================================
    # GROUP WORDS INTO LINES
    # ========================================================

    lines = {}

    all_confidences = []


    for i in range(
        len(data["text"])
    ):

        word = data["text"][i].strip()

        if not word:
            continue


        try:

            confidence = float(
                data["conf"][i]
            )

        except:

            confidence = -1


        # Ignore weak OCR detections
        if confidence < 40:
            continue


        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )


        if key not in lines:

            lines[key] = []


        lines[key].append(
            (
                data["left"][i],
                word,
                confidence
            )
        )


    # ========================================================
    # BUILD COMPLETE LINES
    # ========================================================

    output_lines = []


    for key in sorted(
        lines.keys()
    ):

        words = lines[key]


        # Left-to-right
        words.sort(
            key=lambda x: x[0]
        )


        line_words = []

        line_confidences = []


        for (
            x,
            word,
            confidence
        ) in words:

            line_words.append(
                word
            )

            line_confidences.append(
                confidence
            )


        line = " ".join(
            line_words
        )


        line = clean_text(
            line
        )


        if not is_valid_text(
            line
        ):
            continue


        output_lines.append(
            line
        )


        all_confidences.extend(
            line_confidences
        )


    # ========================================================
    # IF NOTHING FOUND
    # ========================================================

    if not output_lines:

        return (
            "No clear text detected",
            0
        )


    # ========================================================
    # COMPLETE OCR TEXT
    # ========================================================

    final_text = "\n".join(
        output_lines
    )


    final_text = clean_text(
        final_text
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if all_confidences:

        confidence = (
            sum(all_confidences)
            /
            len(all_confidences)
        )

    else:

        confidence = 0


    return (
        final_text,
        confidence
    )


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_reader():

    global latest_frame
    global running


    while running:

        ret, frame = cap.read()


        if ret:

            with frame_lock:

                latest_frame = frame

        else:

            time.sleep(
                0.01
            )


# ============================================================
# OCR THREAD
# ============================================================

def ocr_worker():

    global ocr_text
    global ocr_confidence
    global running


    last_good_text = ""


    while running:

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()


        if frame is None:

            time.sleep(
                0.05
            )

            continue


        try:

            text, confidence = perform_ocr(
                frame
            )


            if (
                text
                and
                text != "No clear text detected"
                and
                confidence >= 45
            ):

                # Avoid replacing a good result
                # with a very short bad result.

                if (
                    len(last_good_text) > 100
                    and
                    len(text)
                    <
                    len(last_good_text) * 0.30
                ):

                    pass

                else:

                    with ocr_lock:

                        ocr_text = text

                        ocr_confidence = confidence


                    last_good_text = text


        except Exception as e:

            print(
                "OCR error:",
                e
            )


        # OCR interval
        time.sleep(
            0.35
        )


# ============================================================
# TTS STATUS
# ============================================================

def set_tts_status(status):

    global tts_status

    with tts_status_lock:

        tts_status = status


def get_tts_status():

    with tts_status_lock:

        return tts_status


# ============================================================
# REQUEST TTS
# ============================================================

def request_tts(text):

    global last_spoken_text


    if not text:
        return


    if (
        text == "Looking for text..."
        or
        text == "No clear text detected"
    ):

        set_tts_status(
            "No text"
        )

        return


    # Replace OCR line breaks with spaces
    speech_text = text.replace(
        "\n",
        " "
    )


    speech_text = re.sub(
        r"\s+",
        " ",
        speech_text
    )


    speech_text = speech_text.strip()


    if not speech_text:

        return


    # ========================================================
    # CLEAR OLD QUEUED SPEECH
    # ========================================================

    try:

        while True:

            tts_queue.get_nowait()

    except queue.Empty:

        pass


    # ========================================================
    # QUEUE SPEECH
    # ========================================================

    try:

        tts_queue.put_nowait(
            speech_text
        )

    except queue.Full:

        return


# ============================================================
# TTS WORKER
# ============================================================

def tts_worker():

    global running
    global last_spoken_text


    while running:

        try:

            text = tts_queue.get(
                timeout=0.2
            )

        except queue.Empty:

            continue


        if not text:
            continue


        temp_file = None


        try:

            set_tts_status(
                "Generating..."
            )


            tts = gTTS(
                text=text,
                lang="en",
                slow=False
            )


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            ) as fp:

                temp_file = fp.name


            tts.save(
                temp_file
            )


            set_tts_status(
                "Speaking..."
            )


            playsound(
                temp_file
            )


            last_spoken_text = text


            set_tts_status(
                "Ready"
            )


        except Exception as e:

            print(
                "TTS error:",
                e
            )


            set_tts_status(
                "TTS Error"
            )


        finally:

            if temp_file:

                try:

                    os.remove(
                        temp_file
                    )

                except:

                    pass


# ============================================================
# PIXEL-BASED TEXT WRAPPING
# ============================================================

def wrap_text(
    text,
    font,
    font_scale,
    thickness,
    max_width
):

    result = []


    # Preserve OCR lines
    original_lines = text.split(
        "\n"
    )


    for original_line in original_lines:

        original_line = (
            original_line.strip()
        )


        if not original_line:

            continue


        words = original_line.split()


        current_line = ""


        for word in words:

            if not current_line:

                current_line = word

                continue


            test_line = (
                current_line
                +
                " "
                +
                word
            )


            (
                width,
                height
            ), baseline = cv2.getTextSize(
                test_line,
                font,
                font_scale,
                thickness
            )


            if width <= max_width:

                current_line = test_line

            else:

                result.append(
                    current_line
                )

                current_line = word


        if current_line:

            result.append(
                current_line
            )


    return result


# ============================================================
# BUTTON DRAWING
# ============================================================

def draw_button(
    frame,
    x,
    y,
    width,
    height,
    text,
    background,
    border
):

    # Background
    cv2.rectangle(
        frame,
        (x, y),
        (
            x + width,
            y + height
        ),
        background,
        -1
    )


    # Border
    cv2.rectangle(
        frame,
        (x, y),
        (
            x + width,
            y + height
        ),
        border,
        2
    )


    # Button font
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.65

    thickness = 2


    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )


    text_x = (
        x
        +
        (
            width
            -
            text_width
        )
        // 2
    )


    text_y = (
        y
        +
        (
            height
            +
            text_height
        )
        // 2
    )


    cv2.putText(
        frame,
        text,
        (
            text_x,
            text_y
        ),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global button_clicked


    if event != cv2.EVENT_LBUTTONDOWN:

        return


    buttons = param


    # ========================================================
    # SPEAK
    # ========================================================

    sx, sy, sw, sh = buttons["speak"]


    if (
        sx <= x <= sx + sw
        and
        sy <= y <= sy + sh
    ):

        with button_lock:

            button_clicked = "speak"

        return


    # ========================================================
    # CLEAR
    # ========================================================

    cx, cy, cw, ch = buttons["clear"]


    if (
        cx <= x <= cx + cw
        and
        cy <= y <= cy + ch
    ):

        with button_lock:

            button_clicked = "clear"


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = (
    "Smart Real-Time OCR + TTS"
)


#
# IMPORTANT:
# AUTOSIZE prevents the camera/OCR frame from being
# stretched when the window is resized.
#

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_AUTOSIZE
)


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback,
    button_positions
)


# ============================================================
# START CAMERA
# ============================================================

camera_thread = threading.Thread(
    target=camera_reader,
    daemon=True
)

camera_thread.start()


# ============================================================
# START OCR
# ============================================================

ocr_thread = threading.Thread(
    target=ocr_worker,
    daemon=True
)

ocr_thread.start()


# ============================================================
# START TTS
# ============================================================

tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()


# ============================================================
# MAIN DISPLAY LOOP
# ============================================================

while True:

    # ========================================================
    # GET CAMERA FRAME
    # ========================================================

    with frame_lock:

        if latest_frame is None:

            continue


        original_frame = (
            latest_frame.copy()
        )


    # ========================================================
    # ROTATE DISPLAY FRAME
    # ========================================================

    display_frame = rotate_camera(
        original_frame
    )


    # ========================================================
    # IMPORTANT:
    # Resize proportionally for display.
    #
    # This prevents the OCR letters from becoming
    # horizontally stretched.
    # ========================================================

    MAX_DISPLAY_WIDTH = 1400

    MAX_DISPLAY_HEIGHT = 900


    h, w = display_frame.shape[:2]


    scale = min(
        MAX_DISPLAY_WIDTH / w,
        MAX_DISPLAY_HEIGHT / h,
        1.0
    )


    if scale < 1.0:

        display_frame = cv2.resize(
            display_frame,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )


    frame_height = (
        display_frame.shape[0]
    )

    frame_width = (
        display_frame.shape[1]
    )


    # ========================================================
    # OCR RESULT
    # ========================================================

    with ocr_lock:

        current_text = ocr_text

        current_confidence = (
            ocr_confidence
        )


    # ========================================================
    # UI FONT
    # ========================================================

    #
    # SIMPLEX is intentionally used here.
    # It is cleaner and easier to read than DUPLEX.
    #

    OCR_FONT = cv2.FONT_HERSHEY_SIMPLEX

    OCR_FONT_SCALE = 0.48

    OCR_THICKNESS = 1


    # ========================================================
    # OCR PANEL
    # ========================================================

    panel_x = 15

    panel_y = 15

    panel_width = (
        frame_width - 30
    )


    # ========================================================
    # TEXT MARGIN
    # ========================================================

    text_left = (
        panel_x + 25
    )

    text_right = (
        panel_x
        +
        panel_width
        -
        25
    )


    text_width = (
        text_right
        -
        text_left
    )


    # ========================================================
    # WRAP TEXT
    # ========================================================

    wrapped_lines = wrap_text(
        current_text,
        OCR_FONT,
        OCR_FONT_SCALE,
        OCR_THICKNESS,
        text_width
    )


    # ========================================================
    # MAXIMUM DISPLAY LINES
    # ========================================================

    MAX_LINES = 9


    if len(wrapped_lines) > MAX_LINES:

        display_lines = wrapped_lines[
            :MAX_LINES
        ]

        has_more = True

    else:

        display_lines = wrapped_lines

        has_more = False


    # ========================================================
    # LINE SPACING
    # ========================================================

    LINE_HEIGHT = 25


    # ========================================================
    # PANEL HEIGHT
    # ========================================================

    title_space = 48

    status_space = 38

    text_space = (
        len(display_lines)
        *
        LINE_HEIGHT
    )


    panel_height = (
        title_space
        +
        text_space
        +
        status_space
        +
        20
    )


    # Minimum
    panel_height = max(
        panel_height,
        120
    )


    # Maximum
    panel_height = min(
        panel_height,
        int(frame_height * 0.60)
    )


    # ========================================================
    # BLACK PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # OCR RESULT TITLE
    # ========================================================

    cv2.putText(
        display_frame,
        "OCR RESULT",
        (
            text_left,
            panel_y + 32
        ),
        OCR_FONT,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    text_y = (
        panel_y + 62
    )


    for line in display_lines:

        cv2.putText(
            display_frame,
            line,
            (
                text_left,
                text_y
            ),
            OCR_FONT,
            OCR_FONT_SCALE,
            (245, 245, 245),
            OCR_THICKNESS,
            cv2.LINE_AA
        )


        text_y += LINE_HEIGHT


    # ========================================================
    # MORE TEXT
    # ========================================================

    if has_more:

        cv2.putText(
            display_frame,
            "More text...",
            (
                text_left,
                panel_y
                +
                panel_height
                -
                32
            ),
            OCR_FONT,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_text = (
        f"Confidence: {current_confidence:.1f}%"
    )


    cv2.putText(
        display_frame,
        confidence_text,
        (
            text_left,
            panel_y
            +
            panel_height
            -
            13
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # TTS STATUS
    # ========================================================

    current_tts_status = (
        get_tts_status()
    )


    tts_text = (
        f"TTS: {current_tts_status}"
    )


    (
        tts_width,
        tts_height
    ), baseline = cv2.getTextSize(
        tts_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1
    )


    cv2.putText(
        display_frame,
        tts_text,
        (
            frame_width
            -
            tts_width
            -
            25,
            panel_y
            +
            panel_height
            -
            13
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    button_y = (
        panel_y
        +
        panel_height
        +
        12
    )


    button_height = 45


    button_gap = 35


    available_width = (
        frame_width
        -
        50
        -
        button_gap
    )


    button_width = (
        available_width // 2
    )


    # ========================================================
    # SPEAK BUTTON
    # ========================================================

    speak_x = 25


    button_positions["speak"] = (
        speak_x,
        button_y,
        button_width,
        button_height
    )


    draw_button(
        display_frame,
        speak_x,
        button_y,
        button_width,
        button_height,
        "SPEAK",
        (0, 150, 0),
        (0, 255, 0)
    )


    # ========================================================
    # CLEAR BUTTON
    # ========================================================

    clear_x = (
        speak_x
        +
        button_width
        +
        button_gap
    )


    button_positions["clear"] = (
        clear_x,
        button_y,
        button_width,
        button_height
    )


    draw_button(
        display_frame,
        clear_x,
        button_y,
        button_width,
        button_height,
        "CLEAR",
        (75, 75, 75),
        (210, 210, 210)
    )


    # ========================================================
    # HANDLE BUTTON CLICK
    # ========================================================

    with button_lock:

        clicked = button_clicked

        button_clicked = None


    # ========================================================
    # SPEAK CLICKED
    # ========================================================

    if clicked == "speak":

        with ocr_lock:

            text_to_speak = ocr_text


        if (
            text_to_speak
            and
            text_to_speak != "Looking for text..."
            and
            text_to_speak != "No clear text detected"
        ):

            print()
            print(
                "========== SPEAK =========="
            )

            print(
                text_to_speak
            )

            print(
                "==========================="
            )


            request_tts(
                text_to_speak
            )

        else:

            set_tts_status(
                "No text"
            )


    # ========================================================
    # CLEAR CLICKED
    # ========================================================

    elif clicked == "clear":

        with ocr_lock:

            ocr_text = (
                "Looking for text..."
            )

            ocr_confidence = 0


        last_spoken_text = ""


        set_tts_status(
            "Ready"
        )


        print(
            "OCR result cleared."
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        running = False

        break


# ============================================================
# CLEANUP
# ============================================================

running = False

time.sleep(
    0.5
)

cap.release()

cv2.destroyAllWindows()

print()
print(
    "OCR + TTS program stopped."
)