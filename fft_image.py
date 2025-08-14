import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.fft import fft2, fftshift, fftfreq

def generate_2d_gaussian(size=256, sigma=20):
    """Generates a 2D Gaussian blob."""
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return gaussian

def radial_profile(image):
    """Compute the radial average of a 2D image."""
    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

# Parameters
size = 256
sigma = 20
pixel_spacing = 1.0  # size of one pixel (assume 1 unit)

# Generate image
img = generate_2d_gaussian(size=size, sigma=sigma)

# FFT
f_img = fftshift(fft2(img))
power_spectrum = np.abs(f_img)**2

# Wavenumber grid
kx = fftshift(fftfreq(size, d=pixel_spacing))  # cycles per unit length
ky = fftshift(fftfreq(size, d=pixel_spacing))
KX, KY = np.meshgrid(kx, ky)
K_mag = np.sqrt(KX**2 + KY**2)  # wavenumber magnitude

# Radial profile of power spectrum
r_profile = radial_profile(power_spectrum)
radii = np.arange(len(r_profile))
k_max = radii[np.argmax(np.cumsum(r_profile) / np.sum(r_profile) > 0.95)]
lambda_max = size / k_max if k_max > 0 else np.inf

# Plot
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

# Image
ax[0].imshow(img, cmap='viridis', extent=[-size/2, size/2, -size/2, size/2])
ax[0].set_title("2D Gaussian")
ax[0].set_xlabel("x (pixels)")
ax[0].set_ylabel("y (pixels)")

# FFT in wavenumber space
im1 = ax[1].imshow(np.log1p(power_spectrum), cmap='magma',
                   extent=[kx[0], kx[-1], ky[0], ky[-1]])
ax[1].set_title("FFT Power Spectrum (Wavenumber space)")
ax[1].set_xlabel("kₓ (cycles/pixel)")
ax[1].set_ylabel("kᵧ (cycles/pixel)")
fig.colorbar(im1, ax=ax[1], label='log power')

# Radial spectrum
ax[2].plot(radii, r_profile)
ax[2].axvline(k_max, color='red', linestyle='--', label=f"95% energy → λ ≈ {lambda_max:.1f}")
ax[2].set_title("Radial Power Spectrum")
ax[2].set_xlabel("Radial wavenumber (pixels⁻¹)")
ax[2].set_ylabel("Power")
ax[2].legend()

plt.tight_layout()

print(f"Estimated max significant wavelength: λ ≈ {lambda_max:.2f} pixels")


lambda_min = np.pi * sigma
k_max = 1 / lambda_min

from matplotlib.patches import Circle

# FFT wavenumber axes
kx = fftshift(np.fft.fftfreq(size, d=1.0))
ky = fftshift(np.fft.fftfreq(size, d=1.0))
extent = [kx[0], kx[-1], ky[0], ky[-1]]

# FFT power spectrum plot
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(np.log1p(power_spectrum), extent=extent, cmap='magma')
ax.set_title("FFT Power Spectrum (Wavenumber space)")
ax.set_xlabel("kₓ (cycles/pixel)")
ax.set_ylabel("kᵧ (cycles/pixel)")
fig.colorbar(im, ax=ax, label='log power')

# Draw k_max circle
circle = Circle((0, 0), radius=2*k_max, edgecolor='cyan', facecolor='none', linewidth=2, label=f"$k_{{\\max}}$ = {k_max:.4f}")
ax.add_patch(circle)
ax.legend()