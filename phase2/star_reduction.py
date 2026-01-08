from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Load the FITS file
fits_file = '../examples/test_M31_linear.fits'
hdul = fits.open(fits_file)

print(f"Processing: {fits_file}")
hdul.info()

data = hdul[0].data
header = hdul[0].header

# Normalize data for processing
if data.ndim == 3:
    # For color images, work with luminance
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    # Convert to grayscale for star detection
    data_gray = np.mean(data, axis=2)
else:
    data_gray = data.copy()

# Normalize to [0, 255] for OpenCV
data_normalized = (data_gray - data_gray.min()) / (data_gray.max() - data_gray.min())
image_original = (data_normalized * 255).astype('uint8')

# Save original
plt.imsave('./results/image_original.png', data_normalized, cmap='gray')

# Calculate background statistics
mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)

print(f"Fond du ciel: moyenne={mean:.2f}, médiane={median:.2f}, écart-type={std:.2f}")

# Try different threshold if no stars detected
threshold_multiplier = 3.0
daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_multiplier*std)
sources = daofind(data_gray - median)

# If no stars found, try with lower threshold
if sources is None:
    print("Aucune étoile avec seuil=3*std, essai avec seuil=2*std...")
    threshold_multiplier = 2.0
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_multiplier*std)
    sources = daofind(data_gray - median)

# If still no stars, use simple thresholding
if sources is None:
    print("Aucune étoile avec DAOStarFinder, utilisation du seuillage adaptatif...")
    # Use percentile-based thresholding
    threshold_value = np.percentile(data_gray, 99.5)
    mask_temp = (data_gray > threshold_value).astype(np.uint8)
    
    # Find connected components (stars)
    num_labels, labels = cv.connectedComponents(mask_temp)
    sources = []
    
    for label in range(1, num_labels):
        coords = np.where(labels == label)
        if len(coords[0]) > 5:  # Minimum size filter
            y_center = int(np.mean(coords[0]))
            x_center = int(np.mean(coords[1]))
            sources.append({'xcentroid': x_center, 'ycentroid': y_center})

print(f"Nombre d'étoiles détectées: {len(sources) if sources is not None else 0}")

# Create binary star mask
mask = np.zeros_like(data_gray, dtype=np.float32)

if sources is not None:
    for source in sources:
        x, y = int(source['xcentroid']), int(source['ycentroid'])
        # Draw a circle around each star
        cv.circle(mask, (x, y), radius=5, color=1.0, thickness=-1)

# Save binary mask
plt.imsave('./results/masque_etoile.png', mask, cmap='gray')

# Apply Gaussian blur to soften mask edges
mask_blurred = cv.GaussianBlur(mask, (7, 7), 2)

# Normalize mask to [0, 1]
if mask_blurred.max() > 0:
    mask_blurred = mask_blurred / mask_blurred.max()

# Save blurred mask
plt.imsave('./results/masque_flou.png', mask_blurred, cmap='gray')

# Apply erosion to the entire image
kernel = np.ones((5, 5), np.uint8)
image_eroded = cv.erode(image_original, kernel, iterations=2)

# Save eroded image
plt.imsave('./results/image_erodee.png', image_eroded, cmap='gray')

# Combine using mask: I_final = M × I_erode + (1-M) × I_original
# Convert to float for calculation
image_original_f = image_original.astype(np.float32)
image_eroded_f = image_eroded.astype(np.float32)

# Apply the formula
image_final = (mask_blurred * image_eroded_f) + ((1 - mask_blurred) * image_original_f)
image_final = image_final.astype(np.uint8)

# Save final result
plt.imsave('./results/image_finale.png', image_final, cmap='gray')

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image_original, cmap='gray')
axes[0, 0].set_title('Image Originale')
axes[0, 0].axis('off')

axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('Masque Binaire')
axes[0, 1].axis('off')

axes[1, 0].imshow(image_eroded, cmap='gray')
axes[1, 0].set_title('Image Érodée')
axes[1, 0].axis('off')

axes[1, 1].imshow(image_final, cmap='gray')
axes[1, 1].set_title('Résultat Final')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('./results/phase2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nRésultats sauvegardés dans ./results/:")

# Close FITS file
hdul.close()
