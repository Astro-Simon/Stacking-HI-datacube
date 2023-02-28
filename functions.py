import os
import warnings
from astropy.modeling.polynomial import Chebyshev1D
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from photutils.aperture import CircularAperture, aperture_photometry
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy.coordinates import SpectralCoord


def plot_galaxies_positions(data, wcs, list_pixels_X, list_pixels_Y, pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max,
                            X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, flux_units, vmin=None, vmax=None):
    """
    Plot the spatial positions of galaxies.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data array of shape `(n, y, x)` containing the galaxy data.
    wcs : `~astropy.wcs.WCS`
        WCS object containing the world coordinate system information.
    list_pixels_X : `numpy.ndarray`
        Array of shape `(n,)` containing the x-coordinates of the galaxies.
    list_pixels_Y : `numpy.ndarray`
        Array of shape `(n,)` containing the y-coordinates of the galaxies.
    pixel_x_min : `float`
        Minimum x-pixel value.
    pixel_x_max : `float`
        Maximum x-pixel value.
    pixel_y_min : `float`
        Minimum y-pixel value.
    pixel_y_max : `float`
        Maximum y-pixel value.
    X_AR_ini : `float`
        Initial x-coordinate value.
    pixel_X_to_AR : `float`
        Conversion factor between pixel and x-coordinates.
    Y_DEC_ini : `float`
        Initial y-coordinate value.
    pixel_Y_to_Dec : `float`
        Conversion factor between pixel and y-coordinates.
    flux_units : `str`
        Units of flux density.
    vmin : `float`, optional
        Minimum value of the color map. If not provided, the minimum value in the data array is used.
    vmax : `float`, optional
        Maximum value of the color map. If not provided, the maximum value in the data array is used.

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(9, 7.20), subplot_kw={'projection': wcs})

    # Galaxy with integrated flux
    image_int_spectrum = np.zeros((data.shape[1], data.shape[2]))  # Y, X

    for i in range(data.shape[1]):  # Y
        for j in range(data.shape[2]):  # X
            image_int_spectrum[i][j] = np.nansum(data[:, i, j])

    if vmin is None:
        vmin = np.nanmin(image_int_spectrum)
    if vmax is None:
        vmax = np.nanmax(image_int_spectrum)

    galaxia = ax.imshow(image_int_spectrum, cmap="inferno", interpolation='none',
                        extent=[X_AR_ini + pixel_x_min * pixel_X_to_AR, X_AR_ini + pixel_x_max * pixel_X_to_AR,
                                Y_DEC_ini + pixel_y_min * pixel_Y_to_Dec, Y_DEC_ini + pixel_y_max * pixel_Y_to_Dec],
                        aspect='auto', origin="lower", vmin=vmin, vmax=vmax)

    # Plot the colorbar
    cbar = fig.colorbar(galaxia, ticks=np.linspace(vmin, vmax, 6, endpoint=True))
    cbar.ax.set_ylabel('Flux density (%s)' % flux_units, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_offset_text().set_fontsize(14)

    # Plot the galaxy positions
    ax.plot(X_AR_ini + pixel_x_min * pixel_X_to_AR + list_pixels_X * pixel_X_to_AR, 
            Y_DEC_ini + pixel_y_min * pixel_Y_to_Dec + list_pixels_Y * pixel_Y_to_Dec, 'x', ms=7, color='white', 
            transform=ax.transData)

    # Set figure labels and title
    ax.set_xlabel("RA (J2000)", labelpad=1)
    ax.set_ylabel("Dec (J2000)", labelpad=0)
    ax.tick_params(axis='both')
    ax.set_title('Spatial positions of each galaxy')

    # Save the figure
    plt.tight_layout()
    path = '/Verification_process/'
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig("%sgalaxies_positions.pdf" % path)

    print("\nPositions of the galaxies obtained!\n")

def extract_spectrum_from_spatial_circular_region(datacube, wcs, num_channels, center_x, center_y, radius):
    """
    Extract the integrated spectrum from a circular region in a data cube.

    Parameters
    ----------
    data_cube : numpy.ndarray
        The data cube to extract the spectrum from.
    wcs : astropy.wcs.WCS
        The World Coordinate System of the data cube.
    num_channels : int
        The number of channels in the data cube.
    center_x : float
        The x coordinate of the center of the circular region.
    center_y : float
        The y coordinate of the center of the circular region.
    radius : float
        The radius of the circular region.

    Returns
    -------
    integrated_spectrum : numpy.ndarray
        The integrated spectrum of the circular region.
    """
    # Extract spectrum from circular region
    x, y = np.indices(datacube.shape[1:])
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = r <= radius
    integrated_spectrum = datacube[:, mask].sum(axis=1)    
    
    """mask = aperture.to_mask(method='exact')
    
    plt.imshow(mask)
    plt.show()

    plt.imshow(datacube[int(num_channels/2)], cmap='gray_r', origin='lower')
    aperture.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()"""

    return integrated_spectrum

def fit_continuum_of_spectrum(spectrum, x_axis, emission_channel, semirange, degree):
    """
    Fits the continuum of a spectrum and returns the fitted spectrum, fitted central region, and fitted continuum.

    Parameters
    ----------
    spectrum : array_like
        Spectrum to fit continuum of.
    x_axis : array_like
        Spectral axis of the spectrum.
    emission_channel : float
        Central wavelength of the emission line.
    semirange : float
        Half of the width of the region around the emission line.
    degree : int
        Degree of the Chebyshev polynomial to fit the continuum with.

    Returns
    -------
    tuple
        Tuple containing the fitted spectrum, fitted central region, and fitted continuum.
    """

    # Create Spectrum1D object with flux and spectral axis units
    spectrum_object = Spectrum1D(flux=spectrum*u.Jy, spectral_axis=x_axis*u.um)

    # Define region around central emission
    central_emission = SpectralRegion((emission_channel-semirange)*u.um, (emission_channel+semirange+1)*u.um)

    # Fit Chebyshev1D polynomial to the spectrum with and without the central emission region
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        try:
            fitted_model = fit_generic_continuum(spectrum_object, model=Chebyshev1D(degree), exclude_regions=central_emission)
        except:
            fitted_model = fit_generic_continuum(spectrum_object, model=Chebyshev1D(degree))

    # Extract fitted continuum from the fitted model
    fitted_continuum = fitted_model(x_axis*u.um).value

    # Subtract fitted continuum from spectrum to obtain fitted spectrum
    fitted_spectrum = spectrum - fitted_continuum

    # Extract central region of the spectrum and subtract the fitted continuum to obtain fitted central region
    spectrum_central_region = spectrum[emission_channel-semirange:emission_channel+semirange+1]
    fitted_central_region = spectrum_central_region - fitted_model(np.linspace(emission_channel-semirange, emission_channel+semirange, 2*semirange+1)*u.um).value

    return fitted_spectrum, fitted_central_region, fitted_continuum

def plot_spaxel_spectrum(datacube, spaxel_x, spaxel_y, rest_freq, channel_to_freq, num_channels_cubelets,
                          flux_units, num_galaxies, factor=1, name='spectrum_plot'):
    """
    Plot the spectrum of a single spaxel in a datacube.

    Parameters
    ----------
    datacube : `~numpy.ndarray`
        The datacube containing the spectrum.
    spaxel_x : int
        The x-coordinate of the spaxel.
    spaxel_y : int
        The y-coordinate of the spaxel.
    rest_freq : float
        The rest frequency of the spectral line in Hz.
    channel_to_freq : float
        The conversion factor between channel number and frequency in MHz.
    num_channels_cubelets : int
        The number of channels in each cubelet.
    flux_units : str
        The units of the flux density.
    num_galaxies : int, optional
        The number of galaxies to stack, by default 1.
    factor : float, optional
        The factor to scale the spectrum by, by default 1.
    name : str, optional
        The name of the output file, by default 'spectrum_plot.pdf'.

    Returns
    -------
    None

    """
    spectrum = datacube[:, spaxel_y, spaxel_x] * factor #* We extract the spectrum of the spaxel and rescale it

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax2 = ax.twiny()

    ax.set_xlabel("Relative velocity (km/s)")
    ax2.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(f"Flux density ($10^{{-6}}$ {flux_units})", labelpad=12.0)

    freq_axis = np.linspace(rest_freq - channel_to_freq * num_channels_cubelets / 2,
                             rest_freq + channel_to_freq * num_channels_cubelets / 2, num_channels_cubelets) * 1e-6

    vel_axis = SpectralCoord(freq_axis * u.MHz, redshift=0).to(u.km / u.s, doppler_rest=rest_freq * u.Hz,
                                                               doppler_convention='optical')

    ax.grid(True, alpha=0.5, which="minor", ls=":")
    ax.grid(True, alpha=0.7, which="major")

    ax.plot([], [], label=f'Number of cubelets stacked: {num_galaxies}', alpha=0)

    ax.bar(vel_axis, spectrum, color='black', alpha=1, zorder=1, width=20, label='Stacked spectrum')

    ax.invert_xaxis()
    ax.set_ylim(min(spectrum) - 1, max(spectrum) + 1)

    ax.vlines(0, min(spectrum), max(spectrum), linestyles='--', color='red', label='HI position', alpha=1, zorder=0)
    
    # set the legend and layout
    ax.legend(loc='best')
    plt.tight_layout()
    
    # save the figure
    plt.savefig('%s.pdf' % name)