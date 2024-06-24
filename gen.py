from PIL import Image, ImageDraw, ImageFont
import random
import string
import os

# Define the specific characters
characters_mbbank = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'G', 'H', 'K', 'M', 'N', 
    'P', 'Q', 'U', 'V', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'p', 'q', 
    't', 'u', 'v', 'y', 'z'
]

# Function to generate random captcha text using the specified characters
def generate_random_text(length):
    return ''.join(random.choice(characters_mbbank) for _ in range(length))

# Function to generate a captcha image
def generate_captcha_image(text, width=200, height=35):
    # Create a blank image with a transparent background
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Initialize the drawing context
    draw = ImageDraw.Draw(image)

    # Define font sizes and colors
    font_sizes = [30, 32, 34, 36, 38, 40]
    font_colors = [(0, 0, 0), (255, 0, 0), (0, 128, 0), (0, 0, 255), (128, 0, 128)]

    # Load different fonts (update the paths to fonts available on your system)
    font_paths = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/cour.ttf',      # Courier
        'C:/Windows/Fonts/comic.ttf',
        'C:/Windows/Fonts/verdana.ttf'
    ]
    
    # Attempt to load the fonts, skipping any that fail to load
    fonts = []
    for font_path in font_paths:
        for size in font_sizes:
            try:
                fonts.append(ImageFont.truetype(font_path, size))
            except OSError:
                print(f"Could not load font {font_path} with size {size}")

    # Ensure we have at least one font loaded
    if not fonts:
        raise ValueError("No valid fonts could be loaded. Please check the font paths.")

    # Calculate the total width of the text using bounding boxes
    total_text_width = sum(random.choice(fonts).getbbox(char)[2] for char in text)

    # Calculate the starting x-coordinate for placing the characters
    starting_x = (width - total_text_width) // 2

    # Draw each character with a random font, size, and color
    x = starting_x
    for char in text:
        font = random.choice(fonts)
        font_color = random.choice(font_colors)
        bbox = font.getbbox(char)
        y_pos = random.randint(0, max(0, height - bbox[3]))  # Ensure a valid range
        draw.text((x, y_pos), char, font=font, fill=font_color)
        x += bbox[2]

    return image

# Directory to save the images
output_dir = 'captcha_images'
os.makedirs(output_dir, exist_ok=True)

# Generate and save 1000 CAPTCHA images
for _ in range(10000):
    captcha_text = generate_random_text(6)
    image = generate_captcha_image(captcha_text)
    image_path = os.path.join(output_dir, f'{captcha_text}.png')
    image.save(image_path)
    print(f'Generated CAPTCHA image: {image_path}')
