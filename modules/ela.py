from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO


def run_ela(image_path, output_path, quality=90):
    # 1. open the original image, convert to RGB
    with Image.open(image_path) as im_rgba:
        im_rgb = im_rgba.convert(mode='RGB')

    # 2. save it temporarily to a buffer/temp file at the given quality
    #    (this re-compresses it — authentic regions settle, tampered ones don't)
    image_Buffer = BytesIO()
    im_rgb.save(image_Buffer, format="JPEG", quality=quality)

    # 3. open that resaved version
    image1 = Image.open(image_path)
    image_Buffer.seek(0)
    image2 = Image.open(image_Buffer)

    # 4. compute pixel-wise difference between original and resaved
    #    (use ImageChops.difference)
    result = ImageChops.difference(image1, image2)

    # 5. amplify the difference so it's humanly visible
    #    (scale brightness up — a factor like 10-15x)
    enhanced = ImageEnhance.Brightness(result)
    result = enhanced.enhance(15)

    # 6. save the amplified difference image to output_path
    result.save(output_path)

    # 7. return the output path (so main can report it)
    return output_path
