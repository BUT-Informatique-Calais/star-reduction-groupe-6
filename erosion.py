from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Open and read the FITS file
fits_file = './examples/test_M31_linear.fits'
hdul = fits.open(fits_file)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)
    
    # Normalize each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')


# Define kernels and apply erosion
kernel_3x3 = np.ones((3,3), np.uint8)
eroded_3x3_iter1 = cv.erode(image, kernel_3x3, iterations=1)
eroded_3x3_iter3 = cv.erode(image, kernel_3x3, iterations=3)

kernel_5x5 = np.ones((5,5), np.uint8)
eroded_5x5_iter1 = cv.erode(image, kernel_5x5, iterations=1)
eroded_5x5_iter3 = cv.erode(image, kernel_5x5, iterations=3)

kernel_7x7 = np.ones((7,7), np.uint8)
eroded_7x7_iter1 = cv.erode(image, kernel_7x7, iterations=1)
eroded_7x7_iter3 = cv.erode(image, kernel_7x7, iterations=3)
# Save eroded images
if data.ndim == 3:  
    # Color images
    plt.imsave('./results/eroded_3x3_iter1.png', eroded_3x3_iter1)
    plt.imsave('./results/eroded_3x3_iter3.png', eroded_3x3_iter3)
    plt.imsave('./results/eroded_5x5_iter1.png', eroded_5x5_iter1)
    plt.imsave('./results/eroded_5x5_iter3.png', eroded_5x5_iter3)
    plt.imsave('./results/eroded_7x7_iter1.png', eroded_7x7_iter1)
    plt.imsave('./results/eroded_7x7_iter3.png', eroded_7x7_iter3)
else:  
    # Monochrome images
    plt.imsave('./results/eroded_3x3_iter1.png', eroded_3x3_iter1, cmap='gray')
    plt.imsave('./results/eroded_3x3_iter3.png', eroded_3x3_iter3, cmap='gray')
    plt.imsave('./results/eroded_5x5_iter1.png', eroded_5x5_iter1, cmap='gray')
    plt.imsave('./results/eroded_5x5_iter3.png', eroded_5x5_iter3, cmap='gray')
    plt.imsave('./results/eroded_7x7_iter1.png', eroded_7x7_iter1, cmap='gray')
    plt.imsave('./results/eroded_7x7_iter3.png', eroded_7x7_iter3, cmap='gray')

# Close the FITS file
hdul.close()