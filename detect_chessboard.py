import cv2
import numpy as np
import argparse

def wait_key():
    """Waits for the 'q' key to be pressed to close the windows."""
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not load image '{image_path}'.")
    return img

def extrapolate_line(x1, y1, x2, y2, img_shape):
    """Extrapolates a line to the edges of the image."""
    height, width = img_shape[:2]
    
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y0 = int(slope * 0 + intercept)  # Left intersection
        y_max = int(slope * width + intercept)  # Right intersection
        return 0, y0, width, y_max
    else:
        return x1, 0, x2, height  # Vertical line

def calculate_angle(line1, line2):
    """Calculates the angle between two lines."""
    dx1, dy1 = line1[2] - line1[0], line1[3] - line1[1]
    dx2, dy2 = line2[2] - line2[0], line2[3] - line2[1]

    dot_product = dx1 * dx2 + dy1 * dy2
    mag1 = np.sqrt(dx1**2 + dy1**2)
    mag2 = np.sqrt(dx2**2 + dy2**2)

    if mag1 == 0 or mag2 == 0:
        return None

    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    return np.arccos(cos_angle) * (180.0 / np.pi)

def line_intersection(line1, line2):
    """Finds the intersection of two lines if they form an angle close to 90 degrees."""
    angle = calculate_angle(line1, line2)
    
    if angle is None or (85 <= angle <= 95):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        A1, B1, C1 = float(y2 - y1), float(x1 - x2), float((y2 - y1) * x1 + (x1 - x2) * y1)
        A2, B2, C2 = float(y4 - y3), float(x3 - x4), float((y4 - y3) * x3 + (x3 - x4) * y3)

        determinant = A1 * B2 - A2 * B1

        if abs(determinant) >= 1e-10:
            try:
                x = (B2 * C1 - B1 * C2) / determinant
                y = (A1 * C2 - A2 * C1) / determinant
                return int(x), int(y)
            except OverflowError:
                return None
    return None

def detect_intersections(lines, img_shape):
    """Detects intersections between detected lines."""
    intersections = []
    extrapolated_lines = [extrapolate_line(*line[0], img_shape) for line in lines]

    for i in range(len(extrapolated_lines)):
        for j in range(i + 1, len(extrapolated_lines)):
            intersection = line_intersection(extrapolated_lines[i], extrapolated_lines[j])
            if intersection is not None:
                intersections.append(intersection)

    return np.array(intersections)

def find_extreme_points(intersections):
    """Finds the 4 extreme points of the chessboard using the Convex Hull."""
    hull = cv2.convexHull(intersections)
    if len(hull) < 4:
        raise ValueError("Convex Hull does not have enough points to calculate the extremes.")

    hull = np.squeeze(hull)
    rect = np.zeros((4, 2), dtype='float32')

    s = hull.sum(axis=1)
    rect[0] = hull[np.argmin(s)]  # Top-left
    rect[2] = hull[np.argmax(s)]  # Bottom-right

    diff = np.diff(hull, axis=1)
    rect[1] = hull[np.argmin(diff)]  # Top-right
    rect[3] = hull[np.argmax(diff)]  # Bottom-left

    return rect

def calculate_image_size(rect):
    """Calculates the image dimensions based on the 4 extreme points."""
    width_top = np.linalg.norm(rect[0] - rect[1])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_left = np.linalg.norm(rect[0] - rect[3])
    height_right = np.linalg.norm(rect[1] - rect[2])

    width = int((width_top + width_bottom) / 2)
    height = int((height_left + height_right) / 2)

    return width, height

def apply_perspective_transform(img, corners):
    """Applies perspective transformation to the image."""
    width, height = calculate_image_size(corners)
    
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    return cv2.warpPerspective(img, M, (width, height))

def process_image(img):
    """Processes the image, detects lines and intersections, and applies perspective transformation."""
    # Gray scale -> Median blur -> Canny edge detection -> Hough Transform -> Intersections -> Perspective Transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_blurred = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(median_blurred, 30, 120, apertureSize=3, L2gradient=True)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=20)
    if lines is None:
        raise ValueError("No lines detected in the image.")

    intersections = detect_intersections(lines, img.shape)

    if len(intersections) < 4:
        raise ValueError("Less than 4 intersections found. Cannot proceed.")

    rect = find_extreme_points(intersections)
    warped = apply_perspective_transform(img, rect)

    return warped

def save_image(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"Output image saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line detection in a chessboard image.")
    parser.add_argument("image_path", type=str, help="Path to the chessboard image.")
    parser.add_argument("output_path", type=str, help="Path to save the output image.")
    args = parser.parse_args()

    image_path = args.image_path
    output_path = args.output_path

    img = load_image(image_path)
    output = process_image(img)
    save_image(output, output_path)