import time
from PIL import Image

# Open the original image
original_image = Image.open('/home/foysal/Downloads/Test_Stitch_11k.jpg')

# Start timing for compression
start_time = time.time()

# Specify the compression quality (1-95)
compression_quality = 50

# Save the compressed image
original_image.save('/home/foysal/Downloads/Test_Stitch_11k_compressed.jpg', 'JPEG', quality=compression_quality)

# Calculate and print the compression time
compression_time = time.time() - start_time
print(f"Compression Time: {compression_time} seconds")

# Start timing for decompression
start_time = time.time()

# Decompress the image
decompressed_image = Image.open('/home/foysal/Downloads/Test_Stitch_11k_compressed.jpg')

# Calculate and print the decompression time
decompression_time = time.time() - start_time
print(f"Decompression Time: {decompression_time} seconds")
