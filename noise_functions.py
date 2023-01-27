"""def noise_stack_shift(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq):
    #* We start by shifting all the galaxies' positions 10 arcsec (5 px) on the left/right and up/down
    shift = 5 #* Shift in pixels

    #* 1. Shifting positions of the cubelets so they are not centered on galaxies' positions
    list_pixels_X = np.zeros(num_galaxies, int) #* Horizontal pixel position of the galaxies
    list_pixels_Y = np.zeros(num_galaxies, int) #* Vertical pixel position of the galaxies
    for i in range(num_galaxies):
        sign_X = int(random.random()*2)*2-1
        sign_Y = int(random.random()*2)*2-1

        #* Applying the shift
        #!!! WHAT IF THE GALAXY SHIFTED LIE OUT OF THE DATACUBE
        list_pixels_X[i] = int((coords_RA[i]-X_AR_ini)/pixel_X_to_AR) + sign_X*shift
        list_pixels_Y[i] = int((coords_DEC[i]-Y_DEC_ini)/pixel_Y_to_Dec) + sign_Y*shift


    #* We don't touch to the spectral axis: the redshift is the same and so the shift and wrapping processes are the same for each galaxy

    #* Get the cubelets with shifted positions
    noise_cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, list_pixels_X, list_pixels_Y, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units)
    
    #* Now we get the noise cubelets with their spectra shifted to rest-frame and wrapped
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""
    
"""def noise_stack_random(num_galaxies, num_pixels_X, num_pixels_Y, num_channels_cubelets, num_pixels_cubelets, X_AR_ini, Y_DEC_ini, pixel_X_to_AR, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq):
    
    #* 2. Random positions of the datacube so cubelets are not centered on galaxies' positions
    list_pixels_X = np.zeros(num_galaxies, int) #* Right ascension coordinates of each galaxy
    list_pixels_Y = np.zeros(num_galaxies, int) #* Declination coordinates of each galaxy
    for i in range(num_galaxies):
        #* Applying the shift
        #!!! WHAT IF THE GALAXY SHIFTED LIE OUT OF THE DATACUBE
        list_pixels_X[i] = random.uniform(1, num_pixels_X)
        list_pixels_Y[i] = random.uniform(1, num_pixels_Y)

    #* We don't touch to the spectral axis: the redshift is the same and so the shift and wrapping processes are the same for each galaxy
    
    #* Get the cubelets with random positionsnum_pixels_cubelets
    noise_cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, list_pixels_X, list_pixels_Y, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units)
    #* Now we get the noise cubelets with their spectra shifted to rest-frame and wrapped
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""

"""def noise_stack_Healy(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, show_verifications):
    
    Metafunction that extract the cubelets of the data, shift and wrap them and stack them but switching the redshift of each galaxy so we get a noise datacube.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - num_pixels_cubelets [int]: Semirange of spaxels extracted around each galaxy
    - X_AR_ini [float]: Initial value of pixel X (deg)
    - pixel_X_to_AR [float]: Ratio pixel X (pix)/AR (deg)
    - Y_DEC_ini [float]: Initial value of pixel Y (deg)
    - pixel_Y_to_Dec [float]: Ratio pixel Y (pix)/Dec (deg)
    - PSF_datacube [array - float]: Array of the PSF datacube
    - wcs [WCS]: WCS of the file
    - flux_units [string]: Units of the flux
    - redshifts [array - float]: Redshifts of all the galaxies
    - rest_freq [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - central_width [int]: Number of channels where we consider the central (HI) lies; used for calculating sigma
    - show_verifications [bool]: If 'True', plot the spectrum of one of the shifted and wrapped subelets

    • Output
    - stacked_cube [array - float]: Stacked datacube 
    
    
    #* 3. Redshifts switched between galaxies

    #* Get the cubelets with random positions
    noise_cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, False, show_verifications)
    
    #* We randomly distribute the redshifts
    redshifts = random.sample(list(redshifts), len(redshifts))

    #* Now we get the noise cubelets with their spectra shifted to rest-frame and wrapped
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""
