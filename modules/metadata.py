from PIL import Image
from PIL.ExifTags import TAGS


def extract_metadata(image_path):

    # 1. open the image with Pillow
    image = Image.open(image_path)

    # 2. extract raw EXIF data from the image
    exif_data = image._getexif()
    metadata = {}
    if exif_data is not None:
        for tag_id, value in exif_data.items():
            # convert raw tag IDs to human-readable tag names
            tag_name = TAGS.get(tag_id, tag_id)
            metadata[tag_name] = value
            print(f"{tag_name}: {value}")
    else:
        print("No EXIF data found.")

    return metadata
