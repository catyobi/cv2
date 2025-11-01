import cv2
import numpy as np

# Initialize video capture
s = 0
source = cv2.VideoCapture(s)

# Create windows
win_name = 'Mini Photoshop - Video Feed'
control_name = 'Controls'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(control_name, cv2.WINDOW_NORMAL)

# Filter modes
PREVIEW = 0
CANNY = 1
BLUR = 2
GRAYSCALE = 3
SEPIA = 4
SHARPEN = 5
EMBOSS = 6
CARTOON = 7
INVERT = 8
SKETCH = 9

# Current filter
image_filter = PREVIEW

# Trackbars for adjustments
brightness = 0
contrast = 50
saturation = 50

def nothing(x):
    pass

# Create trackbars with better default values
cv2.createTrackbar('Brightness', control_name, 50, 100, nothing)
cv2.createTrackbar('Contrast', control_name, 50, 100, nothing)
cv2.createTrackbar('Blur Level', control_name, 15, 99, nothing)  # Default 15, max 99
cv2.createTrackbar('Edge Threshold', control_name, 80, 255, nothing)

def apply_brightness_contrast(frame, brightness, contrast):
    """Apply brightness and contrast adjustments"""
    b = brightness - 50
    c = contrast / 50.0
    frame = cv2.convertScaleAbs(frame, alpha=c, beta=b)
    return frame

def apply_sepia(frame):
    """Apply sepia filter"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(frame, kernel)

def apply_sharpen(frame):
    """Apply sharpening filter"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)

def apply_emboss(frame):
    """Apply emboss effect"""
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    return cv2.filter2D(frame, -1, kernel)

def apply_cartoon(frame):
    """Apply cartoon effect"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_sketch(frame):
    """Apply pencil sketch effect"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# Instructions panel
instructions = np.zeros((400, 500, 3), dtype=np.uint8)
instructions[:] = (40, 40, 40)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(instructions, 'KEYBOARD SHORTCUTS:', (20, 30), font, 0.6, (255, 255, 255), 2)
cv2.putText(instructions, 'P - Preview (Original)', (20, 70), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'C - Canny Edge Detection', (20, 100), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'B - Blur', (20, 130), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'G - Grayscale', (20, 160), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'S - Sepia', (20, 190), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'H - Sharpen', (20, 220), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'E - Emboss', (20, 250), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'T - Cartoon', (20, 280), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'I - Invert Colors', (20, 310), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'K - Sketch', (20, 340), font, 0.5, (100, 255, 100), 1)
cv2.putText(instructions, 'Q/ESC - Quit', (20, 370), font, 0.5, (255, 100, 100), 1)

cv2.imshow(control_name, instructions)

print("\n=== MINI PHOTOSHOP ===")
print("Keyboard shortcuts:")
print("P - Preview | C - Canny | B - Blur | G - Grayscale")
print("S - Sepia | H - Sharpen | E - Emboss | T - Cartoon")
print("I - Invert | K - Sketch | Q/ESC - Quit")
print("\nUse trackbars to adjust parameters!")
print("- Blur Level: adjusts blur intensity (only works in Blur mode)")
print("- Edge Threshold: adjusts edge sensitivity (only works in Canny mode)")

while True:
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    # Flip video frame
    frame = cv2.flip(frame, 1)
    
    # Get trackbar values
    brightness = cv2.getTrackbarPos('Brightness', control_name)
    contrast = cv2.getTrackbarPos('Contrast', control_name)
    blur_level = cv2.getTrackbarPos('Blur Level', control_name)
    edge_threshold = cv2.getTrackbarPos('Edge Threshold', control_name)
    
    # Ensure blur level is odd and at least 1
    if blur_level < 1:
        blur_level = 1
    if blur_level % 2 == 0:
        blur_level += 1
    
    # Apply selected filter
    if image_filter == PREVIEW:
        result = frame.copy()
    elif image_filter == CANNY:
        # Use the edge_threshold trackbar value
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif image_filter == BLUR:
        # Use the blur_level trackbar value
        result = cv2.GaussianBlur(frame, (blur_level, blur_level), 0)
    elif image_filter == GRAYSCALE:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif image_filter == SEPIA:
        result = apply_sepia(frame)
    elif image_filter == SHARPEN:
        result = apply_sharpen(frame)
    elif image_filter == EMBOSS:
        result = apply_emboss(frame)
    elif image_filter == CARTOON:
        result = apply_cartoon(frame)
    elif image_filter == INVERT:
        result = cv2.bitwise_not(frame)
    elif image_filter == SKETCH:
        result = apply_sketch(frame)
    else:
        result = frame.copy()
    
    # Apply brightness and contrast adjustments (works on all filters)
    result = apply_brightness_contrast(result, brightness, contrast)
    
    # Add filter name overlay
    filter_names = {
        PREVIEW: 'PREVIEW',
        CANNY: f'CANNY EDGES (Threshold: {edge_threshold})',
        BLUR: f'BLUR (Level: {blur_level})',
        GRAYSCALE: 'GRAYSCALE',
        SEPIA: 'SEPIA',
        SHARPEN: 'SHARPEN',
        EMBOSS: 'EMBOSS',
        CARTOON: 'CARTOON',
        INVERT: 'INVERT',
        SKETCH: 'SKETCH'
    }
    
    cv2.putText(result, filter_names.get(image_filter, 'UNKNOWN'), 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow(win_name, result)
    
    # Handle keyboard input
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('G') or key == ord('g'):
        image_filter = GRAYSCALE
    elif key == ord('S') or key == ord('s'):
        image_filter = SEPIA
    elif key == ord('H') or key == ord('h'):
        image_filter = SHARPEN
    elif key == ord('E') or key == ord('e'):
        image_filter = EMBOSS
    elif key == ord('T') or key == ord('t'):
        image_filter = CARTOON
    elif key == ord('I') or key == ord('i'):
        image_filter = INVERT
    elif key == ord('K') or key == ord('k'):
        image_filter = SKETCH

source.release()
cv2.destroyAllWindows()