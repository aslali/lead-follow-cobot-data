from PIL import Image
import os

def batch_crop_images(input_folder, output_folder, crop_box):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image (you can add more image extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Open the image
                image = Image.open(os.path.join(input_folder, file))

                # Crop the image using the specified box (left, upper, right, lower)
                cropped_image = image.crop(crop_box)

                # Save the cropped image to the output folder
                cropped_image.save(os.path.join(output_folder, file))

                print(f"Cropped {file} and saved to {output_folder}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "/home/ali/Downloads/patterns [Autosaved]"
    output_folder = "/home/ali/Downloads/patterns [Autosaved]/new"

    # Specify the crop box as (left, upper, right, lower)
    crop_box = (1, 100, 1280, 620)  # Replace with your desired crop coordinates and dimensions

    # Call the function to batch crop images
    batch_crop_images(input_folder, output_folder, crop_box)