import cv2
import numpy as np


# Function to load and display an image
def load_and_display_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()
    return image


# Function to find and draw contours
def find_and_draw_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Contours", image_contours)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

    return contours


# Function to filter and display contours based on area
def filter_and_display_contours(image, contours, min_area=5000):
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    image_filtered_contours = image.copy()
    cv2.drawContours(image_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)

    cv2.imshow("Filtered Contours", image_filtered_contours)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

    return filtered_contours


# Function to draw each contour in a separate color
def draw_contours_separate_colors(image, contours):
    image_contours_separate_colors = image.copy()
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color
        cv2.drawContours(image_contours_separate_colors, [contour], -1, color, 2)

    cv2.imshow("Contours in Separate Colors", image_contours_separate_colors)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()


# Function to draw bounding shapes for each contour
def draw_bounding_shapes(image, contours):
    if len(contours) > 1:
        contour = contours[1]
        image_bounding_shapes = image.copy()

        # Bounding Rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_bounding_shapes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Bounding Rectangle", image_bounding_shapes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Minimum Area Rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image_bounding_shapes, [box], 0, (0, 0, 255), 2)
        cv2.imshow("Minimum Area Rectangle", image_bounding_shapes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Minimum Enclosing Circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image_bounding_shapes, center, radius, (255, 0, 0), 2)
        cv2.imshow("Minimum Enclosing Circle", image_bounding_shapes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Fit Ellipse
        if len(contour) >= 5:  # FitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(image_bounding_shapes, ellipse, (255, 255, 0), 2)
            cv2.imshow("Fit Ellipse", image_bounding_shapes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Main execution block
def main(image_path):
    image = load_and_display_image(image_path)
    contours = find_and_draw_contours(image)
    filtered_contours = filter_and_display_contours(image, contours)
    draw_contours_separate_colors(image, contours)
    draw_bounding_shapes(image, contours)


if __name__ == "__main__":
    # Replace 'shapes.jpg' with the path to your image file
    main("shapes.jpg")