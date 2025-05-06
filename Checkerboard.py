import cv2
import numpy as np

def save_checkerboard_image(checker_board, square_size_cm):
    rows, cols = 8, 8
    pixels_per_cm = 200  # Adjust this based on your printing resolution

    square_size_px = int(square_size_cm * pixels_per_cm)
    pattern = np.zeros((rows * square_size_px, cols * square_size_px), dtype=np.uint8) # 8 squrares x 200
    # Creating pattern to make white cube 
    for i in range(0, rows, 2): #from 0 to 6
        for j in range(0, cols, 2):
            pattern[i * square_size_px: (i + 1) * square_size_px, j * square_size_px: (j + 1) * square_size_px] = 255

    for i in range(1, rows, 2): # from 1 to 7
        for j in range(1, cols, 2):
            pattern[i * square_size_px: (i + 1) * square_size_px, j * square_size_px: (j + 1) * square_size_px] = 255

    cv2.imwrite(checker_board, pattern)

    # Create a resizable window
    cv2.namedWindow("Checkerboard", cv2.WINDOW_NORMAL)
    # Resize the window (for example, 800x800 pixels)
    cv2.resizeWindow("Checkerboard", 500, 500)

    cv2.imshow("Checkerboard", pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Checkerboard pattern saved to: {checker_board}")

# function Call out
save_checkerboard_image("checkerboard_pattern_8x8.png", square_size_cm=2)
