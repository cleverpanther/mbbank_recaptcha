import cv2
import os
import glob

# Define the directory containing the images
input_dir = r'D:\\Project\\mbbankbizcaptcha\\mbbank_biz_captcha_dataset'
output_dir = r'D:\\Project\\mbbankbizcaptcha\\mbbank_biz_captcha_dataset\\gray'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all .png files in the input directory
image_files = glob.glob(os.path.join(input_dir, '*.png'))

# Process each image file
for image_file in image_files:
    # Read the image
    image = cv2.imread(image_file)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Construct the output file path
    base_name = os.path.basename(image_file)
    gray_image_path = os.path.join(output_dir, base_name)
    
    # Save the grayscale image
    cv2.imwrite(gray_image_path, gray_image)

print(f"Converted {len(image_files)} images to grayscale and saved to {output_dir}")
