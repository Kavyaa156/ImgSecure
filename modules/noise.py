from PIL import Image, ImageEnhance
import numpy as np
import cv2 as cv


def run_noise_analysis(image_path, output_path):
    # 1. open image and convert to grayscale
    src = cv.imread(image_path)
    gray_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 2. apply Gaussian blur to the grayscale image
    blur = cv.GaussianBlur(gray_image, (5, 5), 0)

    # 3. subtract the blurred version from the original grayscale
    diff = cv.absdiff(gray_image, blur)
    diff_image = Image.fromarray(diff)
    # 4. amplify the result so differences are visible
    enhanced = ImageEnhance.Brightness(diff_image)
    result = enhanced.enhance(15)

    # 5. save the noise map to output_path
    result.save(output_path)

    # 6. return output_path
    return output_path
