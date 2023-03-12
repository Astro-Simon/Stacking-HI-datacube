import numpy as np
import os
import csv
from astropy.io import fits
from astropy.wcs import WCS

import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def copy_header(name_orig_cube):
    """
    Copy header from a data cube file to a separate file and extract some useful parameters from the header.

    Parameters
    ----------
    name_orig_cube : str
        Name of the data cube file.

    Returns
    -------
    wcs : astropy.wcs.WCS object
        WCS object used to plot the cube.
    rest_freq : float
        Rest frequency of the data cube.
    pixel_X_to_AR : float
        Pixel scale in the X direction.
    pixel_Y_to_Dec : float
        Pixel scale in the Y direction.
    pixel_scale : float
        Square root of the product of the X and Y pixel scales.
    channel_to_freq : float
        Frequency resolution of the data cube.
    X_AR_ini : float
        Initial X coordinate of the cube in astronomical units.
    Y_DEC_ini : float
        Initial Y coordinate of the cube in declination.
    freq_ini : float
        Initial frequency of the cube.
    freq_final : float
        Final frequency of the cube.
    flux_units : str
        Units of the flux in the data cube.
    """
    # Extract the header
    hdul = fits.open(name_orig_cube)
    hdr = hdul[0].header

    # Create the file of the header and copy the header in it
    path = 'Headers/'
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}header.txt", "w") as f:
        f.write(repr(hdr))

    # Parameters obtained from the header
    pixel_X_to_AR = hdr.get('CDELT1', 0)
    pixel_Y_to_Dec = hdr.get('CDELT2', 0)
    channel_to_freq = hdr.get('CDELT3', 0)
    rest_freq = hdr.get('RESTFRQ')
    if rest_freq is None:
        print("\nWARNING: The rest frequency is missing.")  # Allow user to input values

    X_AR_ini = hdr.get('CRVAL1', 0) - pixel_X_to_AR * hdr.get('CRPIX1', 0)
    X_AR_final = hdr.get('CRVAL1', 0) + pixel_X_to_AR * (hdr.get('NAXIS1', 0) - 1)
    Y_DEC_ini = hdr.get('CRVAL2', 0) - pixel_Y_to_Dec * hdr.get('CRPIX2', 0)
    Y_DEC_final = Y_DEC_ini + pixel_Y_to_Dec * (hdr.get('NAXIS2', 0) - 1)
    freq_ini = hdr.get('CRVAL3', 0) - channel_to_freq * (hdr.get('CRPIX3', 0) - 1)
    freq_final = hdr.get('CRVAL3', 0) - channel_to_freq * hdr.get('CRPIX3', 0) + channel_to_freq * hdr.get('NAXIS3', 0) - 1

    flux_units = hdr.get('BUNIT')

    pixel_scale = np.sqrt(np.abs(pixel_X_to_AR * pixel_Y_to_Dec))

    # Close the header
    hdul.close()

    wcs = WCS(hdr, naxis=[1, 2])

    #!!! What happens if one of these variables are not in the header? Should I create the variable, try to update it if the value is found, and then return it?
    return wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, Y_DEC_ini, freq_ini, freq_final, flux_units

def param_file(filename: str) -> Tuple[str, str, str, str, str, str, int, str, u.Quantity, u.Quantity]:
    """
    Reads an input parameter file and returns a tuple with the parsed parameters.

    Parameters
    ----------
    filename : str
        The path of the input parameter file.

    Returns
    -------
    Tuple[str, str, str, str, str, str, int, str, u.Quantity, u.Quantity]
        A tuple containing the parsed parameters:
            - general_path : str
            - name_orig_data_cube : str
            - name_orig_PSF_cube : str
            - name_catalog : str
            - path_results : str
            - weights_option : str
            - degree_fit_continuum : int
            - bool_calculate_SNR : str
            - semi_distance_around_galaxies : astropy.units.Quantity
            - semi_vel_around_galaxies : astropy.units.Quantity
    """
    input_parameters = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                key, value = line.strip().split("=")
                input_parameters[key] = value

    # Files and paths
    general_path = input_parameters['PATH_FILES']
    name_orig_data_cube = input_parameters['DATA_DATACUBE']
    name_orig_PSF_cube = input_parameters['PSF_DATACUBE']
    name_catalog = input_parameters['CATALOG']
    path_results = input_parameters['PATH_RESULTS']
    os.makedirs(path_results, exist_ok=True)

    # Global parameters
    # Use kpc instead of number of pixels and angstroms/Hz instead of number of channels
    weights_option = input_parameters['WEIGHTS']
    degree_fit_continuum = int(input_parameters['DEGREE_FIT_CONTINUUM'])  # Degree of fit of continuum around emission lines
    bool_calculate_SNR = input_parameters['CALCULATE_SNR']

    # We are going to extract cubelets of 81x81 kpc^2 around each galaxy for data and noise stack
    semi_distance_around_galaxies = float(input_parameters['WIDTH_CUBELETS_KPC']) / 2 * u.kpc

    # Half-range of velocities around the galaxy emission we select and use in the cubelets
    semi_vel_around_galaxies = float(input_parameters['WIDTH_CUBELETS_KMS']) / 2 * u.km / u.s

    return general_path, name_orig_data_cube, name_orig_PSF_cube, name_catalog, path_results, weights_option, degree_fit_continuum, bool_calculate_SNR, semi_distance_around_galaxies, semi_vel_around_galaxies

def data_and_catalog_extraction(name_orig_cube: str, extension: int) -> tuple:
    """
    Extracts relevant data and catalog information from a fits file.
    Parameters:
    -----------
    name_orig_cube : str
        Name of the original fits file.
    extension : int
        Extension of the fits file.

    Returns:
    --------
    tuple
        Tuple containing the following information:
        wcs : astropy.wcs.WCS
            World coordinate system information.
        rest_freq : float
            Rest frequency of the data.
        pixel_X_to_AR : float
            Conversion factor from pixel to right ascension.
        pixel_Y_to_Dec : float
            Conversion factor from pixel to declination.
        pixel_scale : float
            Pixel scale.
        channel_to_freq : float
            Conversion factor from channel to frequency.
        X_AR_ini : float
            Initial right ascension value.
        Y_DEC_ini : float
            Initial declination value.
        freq_ini : float
            Initial frequency value.
        flux_units : str
            Units of flux.
        data : numpy.ndarray
            Extracted data.
        z_min : float
            Minimum accessible redshift.
        z_max : float
            Maximum accessible redshift.
    """
    # Extract header information.
    wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, Y_DEC_ini, freq_ini, freq_final, flux_units = copy_header(name_orig_cube)

    # Extract data information.
    data = fits.getdata(name_orig_cube, ext=extension)
    data = data[0]

    # Determine accessible redshifts.
    z_min = rest_freq / freq_final - 1
    z_max = rest_freq / freq_ini - 1

    return wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, Y_DEC_ini, freq_ini, flux_units, data, z_min, z_max

def get_galaxies_positions(name_catalog:str, z_min:float, z_max:float)->tuple:
    """
    This function selects the spatial coordinates and redshift of the galaxies,
    saves the position of each galaxy using the columns 'RA_08' (column 47) and 'DEC_08' (column 48)
    and the redshift with 'Z_BEST' (column 146).

    Parameters:
    ----------
    name_catalog : str
        The name of the csv catalog containing the RA_08, DEC_08 and Z_BEST columns
    z_min : float
        Minimum redshift value to be considered
    z_max : float
        Maximum redshift value to be considered

    Returns:
    --------
    tuple : (coords_RA, coords_DEC, redshifts, num_galaxies)
        A tuple containing the RA and DEC coordinates and the redshift of each galaxy 
        that satisfies the redshift range (z_min, z_max)
        
    Example:
    --------
    >>> get_galaxies_positions('galaxies.csv', 0.5, 1.0)
    (array([1.2, 4.5, 6.7]), array([2.4, 5.6, 7.8]), array([0.6, 0.8, 0.9]), 3)
    """

    coords_RA = np.zeros(0) #* Right ascension coordinates of each galaxy (in deg)
    coords_DEC = np.zeros(0) #* Declination coordinates of each galaxy (in deg)
    redshifts = np.zeros(0) #* Redshift of each galaxy
    with open(name_catalog, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        num_galaxies = 0 #* We will count the number of galaxies used in the stacking
        for row in reader:
            if(z_min < float(row['Z_BEST']) < z_max): #* We can only use galaxies with spectra redshifted a certain range
                coords_RA = np.append(coords_RA, float(row['RA_08']))
                coords_DEC = np.append(coords_DEC, float(row['DEC_08']))
                redshifts = np.append(redshifts, float(row['Z_BEST']))
                num_galaxies += 1

    return coords_RA, coords_DEC, redshifts, num_galaxies
