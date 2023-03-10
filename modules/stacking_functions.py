from modules.functions import plot_galaxies_positions
from modules.scaling_functions import spatial_scaling, spectral_scaling
import numpy as np
import random
from alive_progress import alive_bar

import numpy as np

def get_cubelets(num_galaxies, redshifts, rest_freq, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, datacube, wcs, flux_units, is_PSF):
    """
    Extracts sub-cubes (cubelets) of a datacube around each galaxy, centered on the galaxy's position and with a given spectral range and spatial size.
    
    Parameters:
    -----------
    - num_galaxies (int): Number of galaxies to extract cubelets for.
    - redshifts (list): List of redshift values for each galaxy.
    - rest_freq (float): Rest frequency of the spectral line.
    - freq_ini (float): Initial frequency of the datacube.
    - channel_to_freq (float): Frequency range per channel in the datacube.
    - num_channels_cubelets (list): List of number of channels to extract for each galaxy.
    - num_pixels_cubelets (list): List of number of pixels to extract for each galaxy.
    - coords_RA (list): List of Right Ascension (RA) coordinates for each galaxy.
    - coords_DEC (list): List of Declination (DEC) coordinates for each galaxy.
    - X_AR_ini (float): Initial position in RA of the datacube.
    - pixel_X_to_AR (float): Conversion factor from pixels to arcseconds in RA.
    - Y_DEC_ini (float): Initial position in DEC of the datacube.
    - pixel_Y_to_Dec (float): Conversion factor from pixels to arcseconds in DEC.
    - datacube (ndarray): 3D numpy array containing the datacube to extract cubelets from.
    - wcs (astropy.wcs.WCS): WCS object for the datacube.
    - flux_units (str): Units of the datacube's flux.
    - is_PSF (bool): If True, the galaxy is assumed to be a Point Spread Function (PSF) and is centered on the datacube.

    Returns:
    -----------
    - cubelets (list): List of 4D numpy arrays containing the cubelets extracted for each galaxy. The dimensions of each array are (number of channels, number of pixels in y, number of pixels in x).

    Raises:
    -----------
    - ValueError: If `num_channels` is 0, indicating that the user provided an invalid value for `semi_vel_around_galaxies`.

    Notes:
    -----------
    - The spectral range of each cubelet is centered on the galaxy's redshifted emission line, with the number of channels given by `num_channels_cubelets`.
    - The spatial size of each cubelet is given by `num_pixels_cubelets` and is centered on the galaxy's position in pixel coordinates, converted from its RA and DEC coordinates using `X_AR_ini`, `pixel_X_to_AR`, `Y_DEC_ini`, and `pixel_Y_to_Dec`.
    - If a galaxy is on the edge of the datacube, spaxels that are out of the boundaries will have null spectra.
    """ 

    cubelets = []
    
    if is_PSF:
        central_spaxel_Y, central_spaxel_X = datacube.shape[1] // 2, datacube.shape[2] // 2
        list_pixels_X = np.repeat(central_spaxel_X, num_galaxies)
        list_pixels_Y = np.repeat(central_spaxel_Y, num_galaxies)
    else:
        list_pixels_X = ((coords_RA - X_AR_ini) // pixel_X_to_AR).astype(int)
        list_pixels_Y = ((coords_DEC - Y_DEC_ini) // pixel_Y_to_Dec).astype(int)
    
    """if show_verifications:
        plot_galaxies_positions(datacube, wcs, list_pixels_X, list_pixels_Y, 1, 1200, 1, 1200, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, flux_units, 12, 12, 15, 0.00005, None, None)"""
        
    for index, z in enumerate(redshifts):
        emission_position = rest_freq / (1 + z)
        channel_emission = int((emission_position - freq_ini) / channel_to_freq)
        num_pixels = num_pixels_cubelets[index]
        num_channels = num_channels_cubelets[index]
        max_range = datacube.shape[0]
        z_min, z_max = channel_emission - num_channels, channel_emission + num_channels + 1
        y_min, y_max = list_pixels_Y[index] - num_pixels, list_pixels_Y[index] + num_pixels + 1
        x_min, x_max = list_pixels_X[index] - num_pixels, list_pixels_X[index] + num_pixels + 1
        
        if num_channels == 0:
            print("\nWon't extract the cubelets (num_channels is null). Check 'semi_vel_around_galaxies'.\n")
            exit()
            
        if z_min >= 0:
            if z_max < max_range:
                cubelet = datacube[z_min:z_max, y_min:y_max, x_min:x_max]
            else:
                cubelet = np.concatenate((datacube[z_min:, y_min:y_max, x_min:x_max], datacube[:z_max-max_range, y_min:y_max, x_min:x_max]))
        else:
            cubelet = np.concatenate((datacube[z_min:, y_min:y_max, x_min:x_max], datacube[:z_max, y_min:y_max, x_min:x_max]))

        if cubelet.size == 0:
            print("\nEmpty cubelet :(")
            print(f"\nWe could not extract the cubelet centered on (x, y) = ({list_pixels_X[index]}, {list_pixels_Y[index]}).\n")
            cubelet = np.zeros((2 * num_pixels + 1, 2 * num_pixels + 1, 2 * num_channels + 1))
            
        cubelets.append(cubelet)
        
    return cubelets

def stacking_process(type_of_datacube, num_galaxies, num_channels_cubelets, num_pixels_cubelets, central_width, pre_stacking_cubelets, weights_option, luminosity_distances):
    
    """
    Stack the input cubelets to produce a single datacube.

    Parameters
    ----------
    type_of_datacube : str
        The type of datacube being stacked.
    num_galaxies : int
        The number of galaxies in the sample.
    num_channels_cubelets : int
        The semirange of channels in spectral axis.
    num_pixels_cubelets : int
        The semirange of spaxels extracted around each galaxy.
    central_width : int
        The number of channels where the central (HI) lies. Used for calculating sigma.
    pre_stacking_cubelets : np.ndarray[float]
        Array of cubelets shifted and wrapped for each galaxy.
    weights_option : str
        The option used to calculate the weights. Available options are 'fabello', 'lah', 'delhaize', and 'None'.
    luminosity_distances : float, optional
        The luminosity distance of the galaxies. Used to calculate Delhaize's weights.

    Returns
    -------
    stacked_cube : np.ndarray[float]
        The stacked datacube.
    """

    stacked_cube = np.zeros((2*num_channels_cubelets + 1, 2*num_pixels_cubelets + 1, 2*num_pixels_cubelets + 1))
    errorbar_min, errorbar_max = 0, 0
    
    with alive_bar((2*num_pixels_cubelets + 1)**2 * num_galaxies, bar='circles', title=f'{type_of_datacube} stacking in progress') as bar:
        noise_evolution = np.zeros(num_galaxies)
        for pixel_Y in range(2*num_pixels_cubelets + 1):
            for pixel_X in range(2*num_pixels_cubelets + 1):
                rescale = 0
                for i in range(num_galaxies):
                    continuum_spectrum = np.concatenate((pre_stacking_cubelets[i, :int(num_channels_cubelets) - central_width, pixel_Y, pixel_X], pre_stacking_cubelets[i, int(num_channels_cubelets) + central_width:, pixel_Y, pixel_X]))

                    RMS = 0
                    for element in continuum_spectrum:
                        RMS += np.sqrt(element**2)

                    RMS /= np.sqrt(len(continuum_spectrum))

                    if weights_option == 'fabello':
                        weight = 1 / RMS**2
                    elif weights_option == 'lah':
                        weight = 1 / RMS
                    elif weights_option == 'delhaize':
                        weight = 1 / (RMS * luminosity_distances[i]**2)**2
                    else:
                        weight = 1

                    rescale += weight
                    stacked_cube[:, pixel_Y, pixel_X] += (pre_stacking_cubelets[i, :, pixel_Y, pixel_X] * weight)
                    
                    if(type_of_datacube=='Data'):
                        if(pixel_X==num_pixels_cubelets):
                            if(pixel_Y==num_pixels_cubelets):
                                continuum_stacked_spectrum = np.concatenate((stacked_cube[:int(num_channels_cubelets) - central_width, pixel_Y, pixel_X], stacked_cube[int(num_channels_cubelets) + central_width:, pixel_Y, pixel_X]))

                                RMS = np.std(continuum_stacked_spectrum)

                                noise_evolution[i] = RMS
                    bar()

                if(type_of_datacube=='Data'):
                    if(pixel_X==num_pixels_cubelets):
                        if(pixel_Y==num_pixels_cubelets):
                            errorbar_min = np.zeros(2*num_channels_cubelets+1)
                            errorbar_max = np.zeros(2*num_channels_cubelets+1)
                            for c in range(2*num_channels_cubelets+1):
                                #print(pre_stacking_cubelets[:, c, pixel_Y, pixel_X], weight)
                                errorbar_min[c] = np.nanmin(pre_stacking_cubelets[:, c, pixel_Y, pixel_X], axis=0) #!!! Should I use the weight?
                                errorbar_max[c] = np.nanmax(pre_stacking_cubelets[:, c, pixel_Y, pixel_X], axis=0)
                else:
                    errorbar_min = np.zeros(2*num_channels_cubelets+1)
                    errorbar_max = np.zeros(2*num_channels_cubelets+1)
                
                stacked_cube[:, pixel_Y, pixel_X] /= rescale
    
    return stacked_cube, errorbar_min, errorbar_max, noise_evolution

def datacube_stack(type_of_datacube, num_galaxies, num_channels_cubelets, num_pixels_cubelets,
                   coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec,
                   datacube, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq,
                   central_width, central_spaxel, central_channel, weights_option, luminosity_distances):
    """
    Process the data cubes for a set of galaxies and stack them.

    Parameters
    ----------
    type_of_datacube : str
        Type of data cube to be processed. Possible values: 'Data', 'Noise' or 'PSF'.
    num_galaxies : int
        Number of galaxies in the sample.
    num_channels_cubelets : list or ndarray
        List with the number of channels for each cubelet.
    num_pixels_cubelets : list or ndarray
        List with the number of pixels for each cubelet.
    coords_RA : list or ndarray
        List with the right ascension coordinate of the center of each cubelet.
    coords_DEC : list or ndarray
        List with the declination coordinate of the center of each cubelet.
    X_AR_ini : float
        Initial pixel value of the X-axis in angle reference frame.
    pixel_X_to_AR : float
        Conversion factor from pixels to angle in the X-axis.
    Y_DEC_ini : float
        Initial pixel value of the Y-axis in angle reference frame.
    pixel_Y_to_Dec : float
        Conversion factor from pixels to angle in the Y-axis.
    datacube : ndarray
        The 3D data cube.
    wcs : WCS object
        World Coordinate System object with the spatial and spectral scales.
    flux_units : str
        Unit of the flux of the datacube.
    redshifts : list or ndarray
        List with the redshift of each galaxy in the sample.
    rest_freq : float
        Rest frequency of the observed line.
    freq_ini : float
        Frequency of the first channel in the data cube.
    channel_to_freq : float
        Conversion factor from channel number to frequency.
    central_width : float
        Width in frequency units of the central part of the observed line.
    spatial_scales : list or ndarray
        List with the spatial scales in the RA and DEC directions of the cubelets.
    spectral_scales : list or ndarray
        List with the spectral scales of the cubelets.
    weights_option : str
        Option to weight the co-addition of the spectra of the cubelets. Possible values: 'uniform' or 'distance'.
    luminosity_distances : list or ndarray
        List with the luminosity distance of each galaxy in the sample.

    Returns
    -------
    stacked_cube : ndarray
        The stacked data cube of the sample.

    """

    # We use the spatial and spectral coordinates to extract cubelets of the galaxies
    if type_of_datacube == 'Noise':
        coords_RA = random.sample(list(coords_RA), len(coords_RA))
        coords_DEC = random.sample(list(coords_DEC), len(coords_DEC))

    coords_RA = np.array(coords_RA)
    coords_DEC = np.array(coords_DEC)

    if type_of_datacube == 'PSF':
        cubelets = get_cubelets(num_galaxies, redshifts, rest_freq, freq_ini,
                                channel_to_freq, num_channels_cubelets,
                                num_pixels_cubelets, None, None, X_AR_ini,
                                pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec,
                                datacube, wcs, flux_units, True)
    else:
        cubelets = get_cubelets(num_galaxies, redshifts, rest_freq, freq_ini,
                                channel_to_freq, num_channels_cubelets,
                                num_pixels_cubelets, coords_RA, coords_DEC,
                                X_AR_ini, pixel_X_to_AR, Y_DEC_ini,
                                pixel_Y_to_Dec, datacube, wcs, flux_units,
                                False)

    num_pixels_cubelets_wanted = int(np.nanmin(num_pixels_cubelets))
    num_channels_cubelets_wanted = int(np.nanmin(num_channels_cubelets))

    spatially_scaled_cubelets = spatial_scaling(num_galaxies, num_pixels_cubelets,
                                                num_pixels_cubelets_wanted, cubelets)
    spatially_spectrally_scaled_cubelets = spectral_scaling(num_galaxies, num_channels_cubelets,
                                                            num_channels_cubelets_wanted,
                                                            num_pixels_cubelets_wanted,
                                                            spatially_scaled_cubelets)

    num_channels_final = int((spatially_spectrally_scaled_cubelets.shape[1] - 1) / 2)
    num_pixels_final = int((spatially_spectrally_scaled_cubelets.shape[2] - 1) / 2)

    stacked_cube, errorbar_min, errorbar_max, noise_evolution = stacking_process(type_of_datacube, num_galaxies, num_channels_final, num_pixels_final,
                                     central_width, spatially_spectrally_scaled_cubelets, weights_option,
                                     luminosity_distances)

    # For the results, we calculate the integrated flux of each cubelet
    integrated_flux_cubelets = np.zeros(num_galaxies)
    if(type_of_datacube == 'Data'):
        for index, cubelet in enumerate(spatially_spectrally_scaled_cubelets):
            integrated_flux_cubelets[index] = np.nansum(cubelet)

    return stacked_cube, integrated_flux_cubelets, errorbar_min, errorbar_max, noise_evolution
