import os
import pytesseract
from PIL import Image

# -------------------------------------------------
# 1. Tesseract location
# -------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -------------------------------------------------
# 2. Image file
# -------------------------------------------------
image_path = "test.jpg"

# Check whether image exists
if not os.path.exists(image_path):
    print("ERROR: test.jpg not found!")
    print("Place test.jpg in the same folder as this Python file.")
    exit()

# -------------------------------------------------
# 3. Open image
# -------------------------------------------------
try:
    image = Image.open(image_path)
    print("Image loaded successfully!")
except Exception as e:
    print("ERROR: Could not open image.")
    print(e)
    exit()

# -------------------------------------------------
# 4. Perform OCR
# -------------------------------------------------
print("\nReading text from image...")

try:
    text = pytesseract.image_to_string(image)

except Exception as e:
    print("ERROR: OCR failed.")
    print(e)
    exit()

# -------------------------------------------------
# 5. Display result
# -------------------------------------------------
print("\n==============================")
print("        OCR RESULT")
print("==============================")

if text.strip():
    print(text)
else:
    print("No text detected.")

print("==============================")