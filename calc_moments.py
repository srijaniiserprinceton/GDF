"""
This script provides the moment calculation for top hat ESA's
"""
import numpy as np
import pyspedas
import cdflib
import xarray as xr
import matplotlib.pyplot as plt


from scipy.integrate import simps
# from calculations.merge_vdf import merge_vdf_data

def calc_moments_delta(vdf, velocity, theta, phi, METHOD='trapz'):
    """
    This function uses the np.trapz function to perform the integration
    
    VDF should have dimension: E_dim, Theta_dim, Phi_dim

    Parameters:
    -----------
    vdf : np.ndarray of dimension (E_dim, Theta_dim, Phi_dim)
        The ion vdf values in s^3/m^-6
    velocity : np.ndarray of dimension E_dim
        The corresponding velocity in m/s. 
        TODO: Convert function to take in energy and do the 
        converion to velocity inside function.
    theta : np.ndarray of dimension Theta_dim
        Defined theta values where VDF is measured
    phi : np.ndarray of dimension Phi_dim
        Defined phi values where VDF is measured

    Returns:
    --------
    density : float
        Zeroth moment of ion VDF corresponding to density
    uvec : 3-element np.ndarray 
        First moment of ion VDF corresopnding to velocity.
    p_mat : np.ndarray of dimension (3 x 3)
        The corresponding pressure tensor. 
    """

    if np.max(theta) > 2. * np.pi:
        theta = np.radians(theta)
    if np.max(phi) > 4. * np.pi:
        phi = np.radians(phi)

    sinT = np.sin(theta)
    cosT = np.cos(theta)

    sinP = np.sin(phi)
    cosP = np.cos(phi)

    if METHOD == 'trapz':
        integral = np.trapz
    if METHOD == 'simps':
        integral = simps

    dphi = np.radians(11.25)
    dtheta = np.radians(11.25)
    # integrate over phi
    density = integral(integral(integral(vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)

    # Convert to Cartesian velocity
    vx = -velocity[:,None,None] * cosP[None, None, :] * sinT[None, :,None]
    vy = -velocity[:,None,None] * sinP[None, None, :] * sinT[None, :,None]
    vz = -velocity[:,None,None]                       * cosT[None, :,None]

    ux = integral(integral(integral(vx * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)/density
    uy = integral(integral(integral(vy * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)/density
    uz = integral(integral(integral(vz * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)/density

    uvec = np.array([ux, uy, uz])

    vx_p = vx - ux
    vy_p = vy - uy
    vz_p = vz - uz

    pxx = integral(integral(integral(vx_p**2 * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)
    pyy = integral(integral(integral(vy_p**2 * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)
    pzz = integral(integral(integral(vz_p**2 * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)

    pxy = integral(integral(integral(vx_p * vy_p * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)
    pxz = integral(integral(integral(vx_p * vz_p * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)
    pyz = integral(integral(integral(vy_p * vz_p * vdf, dx=dphi) * sinT, dx=dtheta) * velocity**2, x=velocity)

    p_mat = np.array([[pxx, pxy, pxz],
                      [pxy, pyy, pyz],
                      [pxz, pyz, pzz]]
    )

    return(density, uvec, p_mat)


def calc_moments(vdf, velocity, theta, phi, METHOD='trapz'):
    """
    This function uses the np.trapz function to perform the integration
    
    VDF should have dimension: E_dim, Theta_dim, Phi_dim

    Parameters:
    -----------
    vdf : np.ndarray of dimension (E_dim, Theta_dim, Phi_dim)
        The ion vdf values in s^3/m^-6
    velocity : np.ndarray of dimension E_dim
        The corresponding velocity in m/s. 
        TODO: Convert function to take in energy and do the 
        converion to velocity inside function.
    theta : np.ndarray of dimension Theta_dim
        Defined theta values where VDF is measured
    phi : np.ndarray of dimension Phi_dim
        Defined phi values where VDF is measured

    Returns:
    --------
    density : float
        Zeroth moment of ion VDF corresponding to density
    uvec : 3-element np.ndarray 
        First moment of ion VDF corresopnding to velocity.
    p_mat : np.ndarray of dimension (3 x 3)
        The corresponding pressure tensor. 
    """

    if np.max(theta) > 2. * np.pi:
        theta = np.radians(theta)
    if np.max(phi) > 4. * np.pi:
        phi = np.radians(phi)

    sinT = np.sin(theta)
    cosT = np.cos(theta)

    sinP = np.sin(phi)
    cosP = np.cos(phi)

    if METHOD == 'trapz':
        integral = np.trapz
    if METHOD == 'simps':
        integral = simps

    # integrate over phi
    density = integral(integral(integral(vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)

    # Convert to Cartesian velocity
    vx = -velocity[:,None,None] * cosP[None, None, :] * sinT[None, :,None]
    vy = -velocity[:,None,None] * sinP[None, None, :] * sinT[None, :,None]
    vz = -velocity[:,None,None]                       * cosT[None, :,None]

    ux = integral(integral(integral(vx * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)/density
    uy = integral(integral(integral(vy * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)/density
    uz = integral(integral(integral(vz * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)/density

    uvec = np.array([ux, uy, uz])

    vx_p = vx - ux
    vy_p = vy - uy
    vz_p = vz - uz

    pxx = integral(integral(integral(vx_p**2 * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)
    pyy = integral(integral(integral(vy_p**2 * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)
    pzz = integral(integral(integral(vz_p**2 * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)

    pxy = integral(integral(integral(vx_p * vy_p * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)
    pxz = integral(integral(integral(vx_p * vz_p * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)
    pyz = integral(integral(integral(vy_p * vz_p * vdf, x=phi) * sinT, x=theta) * velocity**2, x=velocity)

    p_mat = np.array([[pxx, pxy, pxz],
                      [pxy, pyy, pyz],
                      [pxz, pyz, pzz]]
    )

    return(density, uvec, p_mat)


def spher_moments(vdf, velocity, theta, phi):
    """
    Brute force numerical integration. 

    Parameters:
    -----------
    vdf : np.ndarray of dimension (E_dim, Theta_dim, Phi_dim)
        The ion vdf values in s^3/m^6
    velocity : np.ndarray of dimension E_dim
        The corresponding velocity in m/s. 
        TODO: Convert function to take in energy and do the 
        converion to velocity inside function.
    theta : np.ndarray of dimension Theta_dim
        Defined theta values where VDF is measured
    phi : np.ndarray of dimension Phi_dim
        Defined phi values where VDF is measured
    

    Returns:
    --------
    n : float
        Zeroth moment of ion VDF corresponding to density
    uvec : 3-element np.ndarray 
        First moment of ion VDF corresopnding to velocity.
    p_mat : np.ndarray of dimension (3 x 3)
        The corresponding pressure tensor. 
    """
    V = np.array(velocity)

    if np.max(theta) > 2*np.pi:
        theta = np.deg2rad(theta)
    
    if np.max(phi) > 2*np.pi:
        phi = np.deg2rad(phi)

    dlnv   = np.mean(np.diff(np.log(velocity)))
    dtheta = np.mean(np.diff(theta))
    dphi   = np.mean(np.diff(phi))

    sinT = np.sin(theta)
    cosT = np.cos(theta)

    Const = np.ones(vdf.shape[0])*dlnv*dtheta*dphi
    
    sinP = np.sin(phi)
    cosP = np.cos(phi)

    n = np.sum(((vdf*Const[:,None, None])*sinT[None,:,None]*V[:,None,None]**3))
    
    vx = -V[:,None,None] * cosP[None, None, :] * sinT[None, :,None]
    vy = -V[:,None,None] * sinP[None, None, :] * sinT[None, :,None]
    vz = -V[:,None,None]                       * cosT[None, :,None]

    ux = np.nansum(vx*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))/n
    uy = np.nansum(vy*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))/n
    uz = np.nansum(vz*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))/n

    uvec = np.array([ux, uy, uz])

    vx_p = vx - ux
    vy_p = vy - uy
    vz_p = vz - uz

    # On diagonal terms
    pxx  = np.nansum(vx_p**2*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))
    pyy  = np.nansum(vy_p**2*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))
    pzz  = np.nansum(vz_p**2*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))

    # Cross-terms
    pxy = np.nansum(vx_p*vy_p*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))
    pxz = np.nansum(vx_p*vz_p*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))
    pyz = np.nansum(vy_p*vz_p*(Const[:,None, None]*vdf*sinT[None,:,None]*V[:,None,None]**3))

    p_mat = np.array([[pxx, pxy, pxz],
                      [pxy, pyy, pyz],
                      [pxz, pyz, pzz]]
    )

    return(n, uvec, p_mat)

# # NOTE : Below is just a simple testing script for the moment calcuations on MMS.
# if __name__ == "__main__":
#     # Load in the original VDF
#     init_ds   = cdflib.cdf_to_xarray('/home/michael/Research/VDF_paper1_ESA/input_data_files/MMS_2016-01-11_VDFs.cdf', to_datetime=True)

#     time        = init_ds.time.data
#     orig_energy = init_ds.energy.data[700, :, 0, 0]
#     orig_theta  = init_ds.theta.data[700, 0, 0, :]
#     orig_phi    = init_ds.phi.data[700, 0, :, 0]
#     orig_vdf    = init_ds.vdf.data[700, :, :, :] * 1e12

#     tmerge, vdf_merge, energy_merged, vel_merged, theta_merged, phi_merged = merge_vdf_data(
#         time, np.transpose(init_ds.vdf.data, [0, 1, 3, 2]), init_ds.energy.data[:, :, 0, 0], 
#         13.85*np.sqrt(init_ds.energy.data[:, :, 0, 0]), init_ds.theta.data[:, 0, 0, :], init_ds.phi.data[:, 0, :, 0])

#     # Add the extra phi dimension
#     new_phi = np.append(orig_phi, orig_phi[-1] + 11.25)     # This is in degrees.

#     # Add the extra dimension
#     new_vdf              = np.zeros([32, 16, 33])
#     new_vdf[:, :, 0:32]  = np.transpose(orig_vdf, [0, 2, 1])
#     new_vdf[:, :,   32]  = np.transpose(orig_vdf, [0, 2, 1])[:, :, 0]

#     orig_vel    = 13.85*np.sqrt(orig_energy)
#     orig_dlnv   = np.mean(np.diff(np.log(orig_vel)))

#     orig_n, orig_u, orig_p = spher_moments(np.transpose(orig_vdf, [0, 2, 1]), orig_vel * 1000, np.radians(orig_theta), np.radians(orig_phi))

#     simps_n, simps_u, simps_p = calc_moments(new_vdf, orig_vel * 1000, np.radians(orig_theta), np.radians(new_phi))  

    # # Load in the reconstructed VDF data.
    # log_E      = np.load('lnE_mesh.npy')

    # phi_mesh   = np.load('phi_mesh.npy')
    # theta_mesh = np.load('theta_mesh.npy')
    # vdf        = np.load('VDF_3D_rec.npy')
    # vdf        = 10**(vdf)
    
    # vdf_min    = 2.81734931544879e-27
    # vdf_result = vdf * vdf_min * 1e12   # to meters

    # energy = 10.0**(log_E)
    # theta  = theta_mesh[:, 0]
    # phi    = phi_mesh[0, :]

    # velocity = 13.85 * np.sqrt(energy)

    # dlnV   = np.mean(np.diff(np.log(velocity)))
    # dtheta = np.mean(np.diff(theta))
    # dphi   = np.mean(np.diff(phi))

    # n, u, p = spher_moments(vdf_result, velocity * 1000, theta, phi)