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

def str_to_bool(s: str) -> bool:
    """
    Convert a string representation of a boolean to a boolean value.

    Parameters
    ----------
    s : str
        The string to convert.

    Returns
    -------
    bool
        True if the string is 'True', False if the string is 'False'.

    Raises
    ------
    ValueError
        If the string is neither 'True' nor 'False'.

    Examples
    --------
    >>> str_to_bool('True')
    True
    >>> str_to_bool('False')
    False
    >>> str_to_bool('foo')
    Traceback (most recent call last):
        ...
    ValueError: String must be 'True' or 'False'.
    """
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError("String must be 'True' or 'False'.")

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

def extract_spectrum_from_spatial_circular_region(datacube, center_x, center_y, radius):
    """
    Extract the integrated spectrum from a circular region in a data cube.

    Parameters
    ----------
    data_cube : numpy.ndarray
        The data cube to extract the spectrum from.
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

    """aperture = CircularAperture((center_x, center_y), radius)
    mask = aperture.to_mask(method='exact')
    
    plt.imshow(mask)
    plt.show()

    plt.imshow(datacube[int(len(datacube)/2)], cmap='gray_r', origin='lower')
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
    if(fitted_model.c1.value == 0.): #type: ignore
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            fitted_model = fit_generic_continuum(spectrum_object, model=Chebyshev1D(degree))

    fit_curve = fitted_model(x_axis*u.um).value #type: ignore

    # Subtract curve from spectrum to obtain fitted spectrum
    fitted_spectrum = spectrum - fit_curve
    
    fitted_continuum = np.copy(fitted_spectrum)

    fitted_continuum[emission_channel-semirange:emission_channel+semirange+1] = np.nan

    if(np.nanstd(fitted_continuum) == 0 or np.isnan(np.nanstd(fitted_continuum))):
        fitted_continuum = fitted_spectrum

    # Extract central region of the spectrum and subtract the fitted continuum to obtain fitted central region
    fitted_central_region = fitted_spectrum[emission_channel-semirange:emission_channel+semirange+1]

    return fitted_spectrum, fitted_central_region, fitted_continuum
