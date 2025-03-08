import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D

# Create 3D polar grid
phi = np.linspace(0, 2 * np.pi, 50)  # Azimuthal angle
theta = np.linspace(0, np.pi, 50)  # Polar angle

Phi, Theta = np.meshgrid(phi, theta)

# Set radius of core
Rc = 1
# Set radius of beam
Rb = 0.5

# shifting along the y direction (assuming the response function is axisymmetric)
d_xc = 1
d_xb = 0.5

# Convert spherical to Cartesian coordinates (for core)
Xc = Rc * np.sin(Theta) * np.cos(Phi) + d_xc
Yc = Rc * np.sin(Theta) * np.sin(Phi)
Zc = Rc * np.cos(Theta)

# Convert spherical to Cartesian coordinates (for beam)
Xb = Rb * np.sin(Theta) * np.cos(Phi) + d_xb
Yb = Rb * np.sin(Theta) * np.sin(Phi)
Zb = Rb * np.cos(Theta)

# Plot the 3D surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xc, Yc, Zc, color='c', alpha=0.5)
ax.plot_surface(Xb, Yb + 1, Zb + 1, color='r', alpha=0.5)

# Add axes lines along x=0, y=0, z=0
ax.plot([0, 0], [0, 0], [-2.4, 2.4], 'k', linewidth=2)  # z-axis
ax.plot([-2.4, 2.4], [0, 0], [0, 0], 'k', linewidth=2)  # x-axis
ax.plot([0, 0], [-2.4, 2.4], [0, 0], 'k', linewidth=2)  # y-axis

# Set view angle so that x is into the plane
ax.view_init(elev=0, azim=0)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface of a Sphere')
ax.set_aspect('equal')


# now trying to make the same image in a cylindrical polar coordinate
rhog = np.linspace(0,40,50)
phig = np.linspace(0,2*np.pi,20)
zg = np.linspace(-10,50,60)

zm, rhom, phim = np.meshgrid(zg, rhog, phig, indexing='ij')

phi_idx = np.argmin(np.abs(phig - np.pi/2))
alpha = np.radians(60)
d_x = 0

# slicing
# z, rho, phi = zm[:,:,phi_idx], rhom[:,:,phi_idx], phim[:,:,phi_idx]
z, rho, phi = zm, rhom, phim

Z = zm
X = rhom * np.cos(phim)
Y = rhom * np.sin(phim)

r = np.sqrt((rho * np.cos(phi) - d_x)**2 + (rho * np.sin(phi) * np.cos(alpha) - z * np.sin(alpha))**2 +\
            (rho * np.sin(phi) * np.sin(alpha) + z * np.cos(alpha))**2)
cos_theta = (rho * np.sin(phi) * np.sin(alpha) + z * np.cos(alpha)) / r
tan_phi_num = (rho * np.sin(phi) * np.cos(alpha) - z * np.sin(alpha))
tan_phi_denom = (rho * np.cos(phi) - d_x)

fig = plt.figure(figsize=(8, 6))
plt.contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], r[:,:,phi_idx], levels=20)
plt.colorbar()
plt.scatter(Z[:,:,phi_idx], Y[:,:,phi_idx], color='k', s=1)
plt.gca().set_aspect('equal')

fig = plt.figure(figsize=(8, 6))
plt.contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], np.arccos(cos_theta)[:,:,phi_idx] * 180 / np.pi, levels=20)
plt.colorbar()
plt.scatter(Z[:,:,phi_idx], Y[:,:,phi_idx], color='k', s=1)
plt.gca().set_aspect('equal')

fig = plt.figure(figsize=(8, 6))
plt.contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], np.arctan2(tan_phi_denom, tan_phi_num)[:,:,phi_idx] * 180 / np.pi, levels=20)
plt.colorbar()
plt.scatter(Z[:,:,phi_idx], Y[:,:,phi_idx], color='k', s=1)
plt.gca().set_aspect('equal')

# making an interactive plotter for taking 2D cross-sections in the X, Y, Z dimensions
from matplotlib.widgets import Slider, Button, RadioButtons

axis_color = 'lightgoldenrodyellow'

fig, ax = plt.subplots(2,3,figsize=(8,8))

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.15, wspace=0.1, hspace=0.1)

z_idx = 10
rho_idx = 10
phi_idx = np.argmin(np.abs(phig - np.pi/2))

# Draw the initial plot
im1 = ax[0,0].contourf(X[z_idx,:,:], Y[z_idx,:,:], r[z_idx,:,:], levels=20)
im2 = ax[0,1].contourf(X[z_idx,:,:], Y[z_idx,:,:], np.arccos(cos_theta)[z_idx,:,:] * 180 / np.pi, levels=20)
im3 = ax[0,2].contourf(X[z_idx,:,:], Y[z_idx,:,:], np.arctan2(tan_phi_denom, tan_phi_num)[z_idx,:,:] * 180 / np.pi, levels=20)

# im4 = ax[1,0].contourf(X[:,rho_idx,:], Z[:,rho_idx,:], r[:,rho_idx,:], levels=20)
# im5 = ax[1,1].contourf(X[:,rho_idx,:], Z[:,rho_idx,:], np.arccos(cos_theta)[:,rho_idx,:] * 180 / np.pi, levels=20)
# im6 = ax[1,2].contourf(X[:,rho_idx,:], Z[:,rho_idx,:], np.arctan2(tan_phi_denom, tan_phi_num)[:,rho_idx,:] * 180 / np.pi, levels=20)

im7 = ax[1,0].contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], r[:,:,phi_idx], levels=20)
im8 = ax[1,1].contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], np.arccos(cos_theta)[:,:,phi_idx] * 180 / np.pi, levels=20)
im9 = ax[1,2].contourf(Z[:,:,phi_idx], Y[:,:,phi_idx], np.arctan2(tan_phi_denom, tan_phi_num)[:,:,phi_idx] * 180 / np.pi, levels=20)

for axs in ax.flatten():
    axs.set_aspect('equal')

# Add three sliders for tweaking the parameters

# Define an axes area and draw a slider in it
z_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
z_slider = Slider(z_slider_ax, 'z_slice', zg[0], zg[-1], valinit=10)

# # Draw another slider
# rho_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
# rho_slider = Slider(rho_slider_ax, 'rho_slice', rhog[0], rhog[-1], valinit=10)

# Draw another slider
phi_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
phi_slider = Slider(phi_slider_ax, 'phi_slice', phig[0], phig[-1], valinit=np.pi/2)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    for axs in ax.flatten():
        axs.clear()

    z_slice, phi_slice = z_slider.val, phi_slider.val

    z_slice_idx = np.argmin(np.abs(zg - z_slice))
    # rho_slice_idx = np.argmin(np.abs(rhog - rho_slice))
    phi_slice_idx = np.argmin(np.abs(phig - phi_slice))

    ax[0,0].contourf(X[z_slice_idx,:,:], Y[z_slice_idx,:,:], r[z_slice_idx,:,:], levels=20)
    ax[0,1].contourf(X[z_slice_idx,:,:], Y[z_slice_idx,:,:], np.arccos(cos_theta)[z_slice_idx,:,:] * 180 / np.pi, levels=20)
    ax[0,2].contourf(X[z_slice_idx,:,:], Y[z_slice_idx,:,:], np.arctan2(tan_phi_denom, tan_phi_num)[z_slice_idx,:,:] * 180 / np.pi, levels=20)

    # ax[1,0].contourf(X[:,rho_slice_idx,:], Z[:,rho_slice_idx,:], r[:,rho_slice_idx,:], levels=20)
    # ax[1,1].contourf(X[:,rho_slice_idx,:], Z[:,rho_slice_idx,:], np.arccos(cos_theta)[:,rho_slice_idx,:] * 180 / np.pi, levels=20)
    # ax[1,2].contourf(X[:,rho_slice_idx,:], Z[:,rho_slice_idx,:], np.arctan2(tan_phi_denom, tan_phi_num)[:,rho_slice_idx,:] * 180 / np.pi, levels=20)

    ax[1,0].contourf(Z[:,:,phi_slice_idx], Y[:,:,phi_slice_idx], r[:,:,phi_slice_idx], levels=20)
    ax[1,1].contourf(Z[:,:,phi_slice_idx], Y[:,:,phi_slice_idx], np.arccos(cos_theta)[:,:,phi_slice_idx] * 180 / np.pi, levels=20)
    ax[1,2].contourf(Z[:,:,phi_slice_idx], Y[:,:,phi_slice_idx], np.arctan2(tan_phi_denom, tan_phi_num)[:,:,phi_slice_idx] * 180 / np.pi, levels=20)

    fig.canvas.draw_idle()

z_slider.on_changed(sliders_on_changed)
# rho_slider.on_changed(sliders_on_changed)
phi_slider.on_changed(sliders_on_changed)



