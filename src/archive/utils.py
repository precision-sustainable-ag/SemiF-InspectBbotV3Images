import os
import rawpy
import imageio
import cv2
import numpy as np
import datetime
import rawpy
import pytz
from pprint import pprint
from sklearn.linear_model import LinearRegression
from pathlib import Path
import exifread
from typing import Optional
from dataclasses import dataclass

@dataclass
class ImageMetadata:
    ImageWidth: int
    ImageLength: int
    BitsPerSample: int
    Compression: int
    PhotometricInterpretation: int
    Make: str
    Model: str
    Orientation: int
    SamplesPerPixel: int
    XResolution: str
    YResolution: str
    PlanarConfiguration: int
    ResolutionUnit: int
    Software: str
    DateTime: str
    Rating: int
    ExifOffset: int
    ExposureTime: str
    FNumber: str
    ExposureProgram: int
    ISOSpeedRatings: int
    RecommendedExposureIndex: int
    ExifVersion: list
    DateTimeOriginal: str
    DateTimeDigitized: str
    BrightnessValue: str
    ExposureBiasValue: str
    MaxApertureValue: str
    MeteringMode: int
    LightSource: int
    Flash: int
    FocalLength: str
    FileSource: int
    SceneType: int
    CustomRendered: int
    ExposureMode: int
    WhiteBalance: int
    DigitalZoomRatio: str
    FocalLengthIn35mmFilm: int
    SceneCaptureType: int
    Contrast: int
    Saturation: int
    Sharpness: int
    LensModel: str
    LensSpecification: Optional[list] = None
    BodySerialNumber: Optional[str] = None
    MakerNote: Optional[str] = None
    ImageDescription: Optional[str] = None
    UserComment: Optional[str] = None
    ApplicationNotes: Optional[str] = None
    Tag: Optional[int] = None
    SubIFDs: Optional[int] = None

def get_exif(image_path):
        """Creates a dataclass by reading exif metadata, creating a dictionary, and creating dataclass form that dictionary"""
        # Open image file for reading (must be in binary mode)
        f = open(image_path, "rb")
        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        f.close()
        meta = {}
        for x, y in tags.items():
            newval = (
                y.values[0]
                if type(y.values) == list and len(y.values) == 1
                else y.values
            )
            if type(newval) == exifread.utils.Ratio:
                newval = str(newval)
            meta[x.rsplit(" ")[1]] = newval
        # imgmeta = ImageMetadata(**meta)
        return meta

def present_batches(batch_par, last_five=False):
    nc_batches = sorted([x.name for x in batch_par.glob("NC*")])
    md_batches = sorted([x.name for x in batch_par.glob("MD*")])
    
    if last_five:
        batches = md_batches[-5:] + nc_batches[-5:]
    else:
        batches = md_batches + nc_batches

    pprint(batches)

def get_batch_info(batchdir, print_info=True):
    jpgs = sorted([x for x in batchdir.glob(f"*.JPG")])
    raws = sorted([x for x in batchdir.glob("*.ARW")])
    imgset = set([x.stem for x in jpgs])
    rawset = set([x.stem for x in raws])
    if print_info:
        print(f"Difference between number of raws and jpgs: {len(rawset - imgset)}")
        print(f"Number of JPGs: {len(jpgs)}")
        print(f"Number of RAWs: {len(raws)}")
    return jpgs, raws

def convert_epoch_to_edt(epoch_time):
    utc_dt = datetime.datetime.utcfromtimestamp(epoch_time).replace(tzinfo=pytz.utc)
    edt = pytz.timezone('US/Eastern')
    edt_dt = utc_dt.astimezone(edt)
    return edt_dt

def read_raw(path):
    # Read the raw image file
    raw = rawpy.imread(path)
    # Convert the raw image data to a numpy array suitable for display
    rgb = raw.postprocess()
    return rgb

def read_jpg(path):
    jpg = cv2.imread(str(path))
    rgb = cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)
    return rgb

def get_size(path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(path)
    
    # Convert the size to a more readable format (e.g., KB, MB)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if file_size < 1024.0:
            readable_size = f"{file_size:.2f} {unit}"
            break
        file_size /= 1024.0
    return readable_size, file_size

def extract_color_checker(image_path):
    if Path(image_path).name.split(".")[-1] == "ARW":
        # Read the color card image
        raw = rawpy.imread(image_path)
        color_card_rgb = raw.postprocess()
        # Convert the image to grayscale
        gray = cv2.cvtColor(color_card_rgb, cv2.COLOR_RGB2GRAY) 
    elif Path(image_path).name.split(".")[-1] == "JPG":
        bgr = cv2.imread(str(image_path))
        color_card_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(color_card_rgb, cv2.COLOR_BGR2GRAY)
        
    # Use thresholding to find contours
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming there are 24 patches, sort them and extract the colors
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:24]
    # Sort patches from top-left to bottom-right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_contours = [contour for _, contour in sorted(zip(bounding_boxes, contours), key=lambda b: (b[0][1], b[0][0]))]

    # patches = sorted(patches, key=lambda c: cv2.boundingRect(c)[1])
    
    patch_colors = []
    for patch in sorted_contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [patch], -1, 255, -1)
        mean_color = cv2.mean(color_card_rgb, mask=mask)[:3]
        patch_colors.append(mean_color)
    
    # Ensure exactly 24 colors are extracted
    if len(patch_colors) != 24:
        raise ValueError(f"Expected 24 color patches, but found {len(patch_colors)}.")
    
    return np.array(patch_colors)

def calibrate_image(image_path, reference_colors, target_colors):
    raw = rawpy.imread(str(image_path))
    rgb = raw.postprocess()
    # Flatten the image array and convert to float
    rgb_flat = rgb.reshape(-1, 3).astype(np.float32)
    # Fit a linear regression model to map reference colors to target colors
    model = LinearRegression()
    model.fit(reference_colors, target_colors)
    # Apply the model to correct the colors
    corrected_rgb_flat = model.predict(rgb_flat)
    
    # Reshape the corrected flat array back to the image shape
    corrected_rgb = corrected_rgb_flat.reshape(rgb.shape)
    
    # Clip values to valid range and convert to uint8
    corrected_rgb = np.clip(corrected_rgb, 0, 255).astype(np.uint8)
    
    return corrected_rgb
    

def process_images(image_dir, color_card_image_path, output_dir: Path):
    # Extract reference colors from the color card image
    reference_colors = extract_color_checker(color_card_image_path)
    print(reference_colors)
    exit()

    # Standard target colors for X-Rite ColorChecker
    target_colors = np.array([
        [115, 82, 68], 
        [194, 150, 130], 
        [98, 122, 157], 
        [87, 108, 67],
        [133, 128, 177], 
        [103, 189, 170], 
        [214, 126, 44], 
        [80, 91, 166],
        [193, 90, 99], 
        [94, 60, 108], 
        [157, 188, 64], 
        [224, 163, 46],
        [56, 61, 150], 
        [70, 148, 73], 
        [175, 54, 60], 
        [231, 199, 31],
        [187, 86, 149], 
        [8, 133, 161], 
        [243, 243, 242], 
        [200, 200, 200],
        [160, 160, 160], 
        [122, 122, 121], 
        [85, 85, 85], 
        [52, 52, 52]
    ])
    
    # Process each image in the directory
    images = sorted([x for x in Path(image_dir).glob("*.ARW")])
    for image_path in images:
            calibrated_image = calibrate_image(image_path, reference_colors, target_colors)
            # Save the calibrated image as JPEG
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = Path(output_dir, image_path.stem + '_calibrated.jpg')
            imageio.imwrite(output_path, calibrated_image)