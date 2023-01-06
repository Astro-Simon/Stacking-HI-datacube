from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from astropy.modeling import models, fitting
import csv
import random
from photutils.aperture import CircularAperture, aperture_photometry
import os
import warnings

#!!! Can functions call other functions? Metafunctions...

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def copy_header(name_orig_cube):
    """ 
    Function that extract information from the header of the datacube, show it and write it in a file.

    • Input
    - name_orig_cube [string]: name of the datacube with '.fits' extension

    • Output
    - wcs [WCS]: WCS of the file
    - rest_freq_HI [float]: Rest frequency of HI emission Original rest frequency of the datacube
    - pixel_X_to_AR [float]: Ratio pixel X (pix)/AR (deg)
    - pixel_Y_to_Dec [float]: Ratio pixel Y (pix)/Dec (deg)
    - channel_to_freq [float]: Ratio between channel and frequency
    - X_AR_ini [float]: Initial value of pixel X (deg)
    - X_AR_final [float]: Final value of pixel X (deg)
    - Y_DEC_ini [float]: Initial value of pixel Y (deg)
    - Y_DEC_final [float]: Final value of pixel Y (deg)
    - freq_ini [float]: Initial frequency (Hz)
    - freq_final [float]: Final frequency (Hz)
    - flux_units [string]: Units of the flux
    - num_pixels_X [int]: Number of pixels in X axis
    - num_pixels_Y [int]: Number of pixels in Y axis
    - num_channels [int]: Number of channels
    """

    #* We extract the header
    hdul = fits.open(name_orig_cube)

    #* We can show the basic info of the header
    hdul.info()

    #* We assign the variable 'hdr' the header of the datacube
    hdr = hdul[0].header

    #* We use this variable in order to make the plots of the cube, extracting the spatial coordinates only [2, 1]
    wcs = WCS(hdr, naxis=[1, 2])
    
    #* We create the file of the header and copy the header in it
    path = 'Headers/'
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        f = open("%sheader.txt" %path, "w")
        f.close()
    except:
        print("\nThe header was not copied to a file.\n")

    #* Parameters obtained from the header
    #!!! 'try:' can be used in a bloc or do I have to use 1 'try:' per line?
    pixel_X_to_AR = float(hdr['CDELT1'])
    pixel_Y_to_Dec = float(hdr['CDELT2'])
    channel_to_freq = float(hdr['CDELT3'])
    try:
        rest_freq_HI = float(hdr['RESTFRQ'])
    except:
        pass

    X_AR_ini = float(hdr['CRVAL1']) - pixel_X_to_AR*float(hdr['CRPIX1'])
    X_AR_final = float(hdr['CRVAL1']) + pixel_X_to_AR*float(hdr['NAXIS1']-1)
    Y_DEC_ini = float(hdr['CRVAL2']) - pixel_Y_to_Dec*float(hdr['CRPIX2'])
    Y_DEC_final = Y_DEC_ini + pixel_Y_to_Dec*float(hdr['NAXIS2']-1)
    freq_ini = float(hdr['CRVAL3']) - channel_to_freq*float(hdr['CRPIX3']) 
    freq_final = float(hdr['CRVAL3']) - channel_to_freq*float(hdr['CRPIX3']) + channel_to_freq*float(hdr['NAXIS3'])

    flux_units = hdr['BUNIT']

    num_pixels_X = int(hdr['NAXIS1'])
    num_pixels_Y = int(hdr['NAXIS2'])
    num_channels = int(hdr['NAXIS3'])

    #* Show the results on screen
    print("\nNumber of channels: %d" %num_channels)
    print("Units of flux: %s" %hdr['BUNIT'])
    print("Coordinates of first pixel: (AR, Dec) = (%i h %i min %.2f sec, %iº %i' %.2f'')" %(int(X_AR_ini/15), int((X_AR_ini/15-int(X_AR_ini/15))*60), ((X_AR_ini/15-int(X_AR_ini/15))*60-int((X_AR_ini/15-int(X_AR_ini/15))*60))*60, int(Y_DEC_ini), int((Y_DEC_ini-int(Y_DEC_ini))*60), ((Y_DEC_ini-int(Y_DEC_ini))*60-int((Y_DEC_ini-int(Y_DEC_ini))*60))*60))
    print("Coordinates of first pixel: (%.2fº, %.2fº)" %(X_AR_ini, Y_DEC_ini))
    print("Coordinates of last pixel: (%.2fº, %.2fº)" %(X_AR_final, Y_DEC_final))
    print("Initial frequency: %.2f MHz" %(freq_ini/1e6))
    print("Final frequency: %f MHz" %(freq_final/1e6))
    print("Ratio pixel/right ascension: 1 px = %.2eº" %pixel_X_to_AR)
    print("Ratio pixel/declination: 1 px = %.2eº" %pixel_Y_to_Dec)
    print("Ratio channel/frequency: 1 channel = %.2f kHz" %(channel_to_freq/1e3))

    #* We close the header
    hdul.close()

    #!!! What happens if one of these variables are not in the header? Should I create the variable, try to update it if the value is found, and then return it?
    return wcs, rest_freq_HI, pixel_X_to_AR, pixel_Y_to_Dec, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels

def data_and_catalog_extraction(name_orig_cube, extension):
    """
    Metafunction that extract the array of fluxes of the datacube and information of the header.

    • Input
    - name_orig_cube [string]: name of the datacube with '.fits' extension
    - extension [int]: extension of the datacube where fluxed are stocked

    • Output
    - wcs [WCS]: WCS of the file
    - rest_freq_HI [float]: Rest frequency of HI emission Original rest frequency of the datacube
    - pixel_X_to_AR [float]: Ratio pixel X (pix)/AR (deg)
    - pixel_Y_to_Dec [float]: Ratio pixel Y (pix)/Dec (deg)
    - channel_to_freq [float]: Ratio between channel and frequency
    - X_AR_ini [float]: Initial value of pixel X (deg)
    - X_AR_final [float]: Final value of pixel X (deg)
    - Y_DEC_ini [float]: Initial value of pixel Y (deg)
    - Y_DEC_final [float]: Final value of pixel Y (deg)
    - freq_ini [float]: Initial frequency (Hz)
    - freq_final [float]: Final frequency (Hz)
    - flux_units [string]: Units of the flux
    - num_pixels_X [int]: Number of pixels in X axis
    - num_pixels_Y [int]: Number of pixels in Y axis
    - num_channels [int]: Number of channels
    - data [array - float]: Array of fluxed of the datacube
    - z_min [float]: Minimal redshift we have access to with this datacube
    - z_max [float]: Maximal redshift we have access to with this datacube
    """
    
    wcs, rest_freq_HI, pixel_X_to_AR, pixel_Y_to_Dec, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels = copy_header(name_orig_cube)

    data = fits.getdata(name_orig_cube, ext=extension)
    data = data[0]

    #* First we determine which redshifts we have access to in this datacube 
    z_min = rest_freq_HI/freq_final - 1
    z_max = rest_freq_HI/freq_ini - 1

    return wcs, rest_freq_HI, pixel_X_to_AR, pixel_Y_to_Dec, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels, data, z_min, z_max

def get_galaxies_positions(name_catalog, z_min, z_max):
    """
    Function that extract the spatial and spectral positions of the galaxies of the sample.

    • Input
    - name_catalog [string]: name of the catalog of the sample of galaxies
    - z_min [float]: Minimal redshift we have access to with this datacube
    - z_max [float]: Maximal redshift we have access to with this datacube

    • Output
    - coords_RA [array - float]: List of the horizontal coordinates of each galaxy (in deg)
    - coords_DEC [array - float]: List of the vertical coordinates of each galaxy (in deg)
    - redshifts [array - float]: Redshifts of all the galaxies
    - num_galaxies [int]: Number of galaxies of the sample
    """

    #* Now we select the spatial coordinates and the redshift of the galaxies. We save the position of each galaxy using the columns 'RA_08' (column 47) and 'DEC_08' (column 48) and the redshift with 'Z_BEST' (column 146).

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

def galaxia_puntos(data, wcs, list_pixels_X, list_pixels_Y, pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, flux_units, vmin, vmax):
    """
    Function that plot the spatial position of the galaxies of the sample.

    • Input
    - name_orig_cube [string]: name of the datacube with '.fits' extension

    • Output
    - Today, I offer you nothing. Tomorrow? Who knows...
    """
    
    fig, ax = plt.subplots(figsize=(9,7.20), subplot_kw={'projection': wcs})

    #*-------------------------------------- Galaxia con espectro integrado ----------------------------------*#
    #? Plot the data integrating over spectral axis
    image_int_spectrum = np.zeros((data.shape[1], data.shape[2])) #?Y, X

    for i in range(data.shape[1]): #?Y
        for j in range(data.shape[2]): #?X
            image_int_spectrum[i][j] = np.nansum(data[:, i, j])

    if(vmin==None):
        vmin = np.nanmin(image_int_spectrum)
    if(vmax==None):
        vmax = np.nanmax(image_int_spectrum)

    galaxia = ax.imshow(image_int_spectrum, cmap="inferno", interpolation='none', extent=[X_AR_ini + pixel_x_min*pixel_X_to_AR, X_AR_ini + pixel_x_max*pixel_X_to_AR, Y_DEC_ini + pixel_y_min*pixel_Y_to_Dec, Y_DEC_ini + pixel_y_max*pixel_Y_to_Dec], aspect='auto', origin="lower", vmin=vmin, vmax=vmax)

    #?Plot the colorbar
    cbar = fig.colorbar(galaxia, ticks = np.linspace(vmin, vmax, 6, endpoint=True))
    cbar.ax.set_ylabel('Flux density (%s)' %flux_units, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_offset_text().set_fontsize(14)
            
    ax.plot(X_AR_ini + pixel_x_min*pixel_X_to_AR + list_pixels_X*pixel_X_to_AR, Y_DEC_ini + pixel_y_min*pixel_Y_to_Dec + list_pixels_Y*pixel_Y_to_Dec, 'x', ms=7, color='white', transform=ax.transData)

    #?Figure's labels
    ax.set_xlabel("RA (J2000)", labelpad=1)
    ax.set_ylabel("Dec (J2000)", labelpad=0)
    ax.tick_params(axis='both')
    ax.set_title('Spatial positions of each galaxy')
    #?ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    #?ax.xaxis.get_offset_text().set_fontsize(14)

    #?Save the figure
    plt.tight_layout()
    path = '/Verification_process/'
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig("%sgalaxies_positions.pdf" %path)

    print("\nPositions of the galaxies obtained!\n")

def get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, datacube, wcs, flux_units, is_PSF, show_verifications):
    """
    Function that extract cubelets around the galaxies of the data datacube. We extract the whole spectral range.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - num_pixels_cubelets [int]: Semirange of spaxels extracted around each galaxy
    - list_pixels_X [array - int]: Horizontal positions of the galaxies
    - list_pixels_Y [array - int]: Vertical positions of the galaxies
    - X_AR_ini [float]: Initial value of pixel X (deg)
    - pixel_X_to_AR [float]: Ratio pixel X (pix)/AR (deg)
    - Y_DEC_ini [float]: Initial value of pixel Y (deg)
    - pixel_Y_to_Dec [float]: Ratio pixel Y (pix)/Dec (deg)
    - datacube [array - float]: Datacube
    - wcs [WCS]: WCS of the file
    - flux_units [string]: Units of the flux
    - show_verifications [bool]: If 'True', plot the spatial position of each galaxy in the datacube

    • Output
    - cubelets [array - float]: Array of cubelets of each galaxy
    """
    
    if show_verifications:
        galaxia_puntos(datacube, wcs, list_pixels_X, list_pixels_Y, 1, 1200, 1, 1200, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, flux_units, 12, 12, 15, 0.00005, None, None)

    #* First we convert degrees into pixels coordinates
    if(is_PSF == True):
        central_spaxel_Y, central_spaxel_X = datacube.shape[1]/2, datacube.shape[2]/2
        list_pixels_X, list_pixels_Y = np.repeat(central_spaxel_X, num_galaxies), np.repeat(central_spaxel_Y, num_galaxies)
    else:
        list_pixels_X = np.zeros(num_galaxies, int) #* Horizontal pixel position of the galaxies
        list_pixels_Y = np.zeros(num_galaxies, int) #* Vertical pixel position of the galaxies
        for i in range(num_galaxies):
            list_pixels_X[i] = int((coords_RA[i]-X_AR_ini)/pixel_X_to_AR)
            list_pixels_Y[i] = int((coords_DEC[i]-Y_DEC_ini)/pixel_Y_to_Dec)

    #* Now we extract the sub-cubes (cubelets) of num_pixels_cubelets*num_pixels_cubelets px^2 around each galaxy, so the galaxies are spatially centered in their cubelet
    #!!! Have to do something if the galaxy is on the edge of the area of the cube: the spaxels that are out of the boundaries should have null spectra (?)
    cubelets = np.zeros((num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets+1, 2*num_pixels_cubelets+1)) #*Number of galaxies, spectral length, Y length, X length
    for i in range(num_galaxies):
        y_min = int(list_pixels_Y[i]-num_pixels_cubelets)
        y_max = int(list_pixels_Y[i]+num_pixels_cubelets+1)
        x_min = int(list_pixels_X[i]-num_pixels_cubelets)
        x_max = int(list_pixels_X[i]+num_pixels_cubelets+1)
        try:
            cubelets[i] = datacube[:, y_min:y_max, x_min:x_max]
        except:
            cubelets[i] = np.zeros(cubelets[i].shape) 
            
            print("\nWe could not extract the cubelet centered on (x, y) = (%i, %i).\n" %(list_pixels_X[i], list_pixels_Y[i]))

    return cubelets

def shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, emission_channel, num_channels_cubelets, num_pixels_cubelets, cubelets):
    """
    Function that shifts the spectral axis of cubelets around the frequency of interest (HI - 1420 MHz) and then wrap the part of the spectrum that is out of boundaries.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - redshifts [array - float]: Redshifts of all the galaxies
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - num_channels_cubelets [int]: Number of channels
    - num_pixels_cubelets [int]: Semirange of spaxels extracted around each galaxy
    - cubelets [array - float]: Array of cubelets of each galaxy

    • Output
    - shifted_wrapped_cubelets [array - float]: Array of cubelets shifted and wrapped of each galaxy
    """

    #* We start by finding for each spectrum the position of their HI line. We want to place each spectrum centered on its HI line. In order to do that we have to calculate on which channel this line is and make it the centered channel of the stacked spectrum
    shifted_wrapped_cubelets = np.zeros((num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets+1, 2*num_pixels_cubelets+1))
    for index, redshift in enumerate(redshifts):
        HI_positions = rest_freq_HI/(1+redshift) #* Frequency of the position of the HI line (spectral observational position - not at rest)
        channels_HI = int((HI_positions - freq_ini)/channel_to_freq) #* Channel of the HI line (spectral observational position - not at rest)
        shift_in_channels = emission_channel - channels_HI #* Number of channels that we have to shift the spectrum in order to center it (-: left, +: right)
        for pixel_Y in range(2*num_pixels_cubelets+1):
            for pixel_X in range(2*num_pixels_cubelets+1):
                if(shift_in_channels < 0): #* Shift to the left + wrap
                    shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X] = np.concatenate((cubelets[index, abs(shift_in_channels):, pixel_Y, pixel_X], cubelets[index, :abs(shift_in_channels), pixel_Y, pixel_X]))

                elif(shift_in_channels > 0): #* Shift to the right + wrap
                    shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X] = np.concatenate((cubelets[index, (num_channels_cubelets-shift_in_channels):, pixel_Y, pixel_X], cubelets[index, :(num_channels_cubelets-shift_in_channels), pixel_Y, pixel_X]))

                else: #* No shift nor wrap
                    shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X] = cubelets[index, :, pixel_Y, pixel_X]

    return shifted_wrapped_cubelets

def stacking_process(num_galaxies, num_channels_cubelets, num_pixels_cubelets, central_width, shifted_wrapped_cubelets):
    """
    Function that stack the subelets.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - num_pixels_cubelets [int]: Semirange of spaxels extracted around each galaxy
    - central_width [int]: Number of channels where we consider the central (HI) lies; used for calculating sigma
    - shifted_wrapped_cubelets [array - float]: Array of cubelets shifted and wrapped of each galaxy

    • Output
    - stacked_cube [array - float]: Stacked datacube 
    """

    stacked_cube = np.zeros((num_channels_cubelets, 2*num_pixels_cubelets+1, 2*num_pixels_cubelets+1))
    
    for pixel_Y in range(2*num_pixels_cubelets+1):
        for pixel_X in range(2*num_pixels_cubelets+1):
            rescale = 0
            for i in range(num_galaxies):
                continuum_spectrum = np.concatenate((shifted_wrapped_cubelets[i, :int(num_channels_cubelets/2)-central_width, pixel_Y, pixel_X], shifted_wrapped_cubelets[i, int(num_channels_cubelets/2)+central_width:, pixel_Y, pixel_X])) #* We use the continuum in order to calculate sigma and use it in the weights

                sigma = np.std(continuum_spectrum)
                weight = 1/np.sqrt(sigma)
                rescale += weight
                stacked_cube[:, pixel_Y, pixel_X] += shifted_wrapped_cubelets[i, :, pixel_Y, pixel_X]*weight
            #* We divide the integrated flux by the number of galaxies 
            stacked_cube[:, pixel_Y, pixel_X] /= rescale
            #print(stacked_cube[:, pixel_Y, pixel_X])
    
    """for i in range(num_galaxies):
        for y in range(stacked_cube.shape[1]):
            for x in range(stacked_cube.shape[2]):
                if max(stacked_cube[:, y, x]) > 10:
                    print(x, y, np.argmax(stacked_cube[:, y, x]), max(stacked_cube[:, y, x]))
    exit()"""
    
    return stacked_cube

def datacube_stack(type_of_datacube, num_galaxies, num_channels_cubelets, num_pixels_cubelets, emission_channel, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, datacube, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, central_width, show_verifications):
    """
    Metafunction that extract the cubelets, shift and wrap them and stack them.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - num_pixels_cubelets [int]: Semirange of spaxels extracted around each galaxy
    - coords_RA [array - float]: List of the horizontal coordinates of each galaxy (in deg)
    - coords_DEC [array - float]: List of the vertical coordinates of each galaxy (in deg)
    - X_AR_ini [float]: Initial value of pixel X (deg)
    - pixel_X_to_AR [float]: Ratio pixel X (pix)/AR (deg)
    - Y_DEC_ini [float]: Initial value of pixel Y (deg)
    - pixel_Y_to_Dec [float]: Ratio pixel Y (pix)/Dec (deg)
    - data [array - float]: Array of the data datacube
    - wcs [WCS]: WCS of the file
    - flux_units [string]: Units of the flux
    - redshifts [array - float]: Redshifts of all the galaxies
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - central_width [int]: Number of channels where we consider the central (HI) lies; used for calculating sigma
    - show_verifications [bool]: If 'True', plot the spectrum of one of the shifted and wrapped subelets

    • Output
    - stacked_cube [array - float]: Stacked datacube 
    """

    #* We use the spatial coordinates to determine which spaxels contain a galaxy.

    if(type_of_datacube == 'PSF'):
        cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, datacube, wcs, flux_units, True, show_verifications)
    else:
        cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, datacube, wcs, flux_units, False, show_verifications)

    #* Now we shift each spectrum in each subcube to place it in rest frame with its HI emission at central channel
    #!!! Possible problem: if some cubelets have same spaxels (don't know if it's an issue)
    #!!! How do we calculate the noises now?

    if(type_of_datacube == 'Noise'):
        redshifts = random.sample(list(redshifts), len(redshifts))

    shifted_wrapped_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, emission_channel, num_channels_cubelets, num_pixels_cubelets, cubelets)

    """
    if(show_verifications):
        index, pixel_Y, pixel_X = 32, 6, 9
        fig, ax = plt.subplots(figsize=(19.2, 10.8))

        ax.plot(np.linspace(1, num_channels_cubelets, num_channels_cubelets), shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X]+np.ones(num_channels_cubelets)*0.0005, label="shifted part")
        ax.plot(np.linspace(1, num_channels_cubelets, num_channels_cubelets), cubelets[index, :, pixel_Y, pixel_X], label="original")
        ax.vlines(hi_position, min(cubelets[index, :, pixel_Y, pixel_X]), max(cubelets[index, :, pixel_Y, pixel_X]), linestyles='dashdot', color='red', label='HI position',  alpha=1, zorder=0)
        ax.vlines(hi_position+my_shift, min(shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X])+0.0005, max(shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X])+0.0005, linestyles='dashdot', color='red', alpha=1, zorder=0)
        if(my_shift>0):
            ax.fill_between(np.linspace(0, abs(my_shift), 1000), min(cubelets[32, :, 6, 9])+0.0005, max(cubelets[32, :, 6, 9])+0.0005, color='green', alpha=0.25, label="Shift=%i" %my_shift)
            ax.fill_between(np.linspace(num_channels_cubelets - abs(my_shift), num_channels_cubelets, 1000), min(shifted_wrapped_cubelets[32, :, 6, 9]), max(shifted_wrapped_cubelets[32, :, 6, 9]), color='green', alpha=0.25)
        else:
            ax.fill_between(np.linspace(0, abs(my_shift), 1000), min(cubelets[index, :, pixel_Y, pixel_X]), max(cubelets[index, :, pixel_Y, pixel_X]), color='green', alpha=0.25, label="Shift=%i" %my_shift)
            ax.fill_between(np.linspace(num_channels_cubelets - abs(my_shift), num_channels_cubelets, 1000), min(shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X])+0.0005, max(shifted_wrapped_cubelets[index, :, pixel_Y, pixel_X])+0.0005, color='green', alpha=0.25)
        ax.legend(loc='best')

        #?Figure's labels
        ax.set_xlabel("Channels", labelpad=1)
        ax.set_ylabel("Relative flux", labelpad=0)
        ax.set_title('Shift and wrapping processes example')
        ax.tick_params(axis='y', labelsize=0, length=0)

        #?Save the figure
        plt.tight_layout()
        path = '/Verification_process/'
        if not os.path.isdir(path):
            os.makedirs(path)

        plt.savefig("%sshift_wrap_process.pdf" %path) #?Guardamos la imagen
    """

    #!!! CORRECTIONS TO THIS PREVIOUS PART MUST BE ADDED

    #* The number of channels for the new spectrum will be 242*2, which is the maximum number of channels that can be necessary for co-adding the spectra (supposing HI line in channel 1 and in channel 242 for two different spectra).

    stacked_cube = stacking_process(num_galaxies, num_channels_cubelets, num_pixels_cubelets, central_width, shifted_wrapped_cubelets)

    return stacked_cube

"""def PSF_stack(num_galaxies, num_channels_cubelets, num_pixels_cubelets, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF_datacube, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, show_verifications):
    
    Metafunction that extract the cubelets of the PSF, shift and wrap them and stack them.

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
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - central_width [int]: Number of channels where we consider the central (HI) lies; used for calculating sigma
    - show_verifications [bool]: If 'True', plot the spectrum of one of the shifted and wrapped subelets

    • Output
    - stacked_cube [array - float]: Stacked datacube 
    

    #* We don't use any spatial coordinates because we only need the center of the datacube.
    cubelets = get_cubelets(num_galaxies, num_channels_cubelets, num_pixels_cubelets, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF_datacube, wcs, flux_units, True, show_verifications)

    #* Now we shift each spectrum in each subcube to place it in rest frame with its HI emission at central channel
    #!!! Possible problem: if some cubelets have same spaxels (don't know if it's an issue)
    #!!! How do we calculate the noises now?
    shifted_wrapped_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, cubelets)

    #* The number of channels for the new spectrum will be 242*2, which is the maximum number of channels that can be necessary for co-adding the spectra (supposing HI line in channel 1 and in channel 242 for two different spectra).

    stacked_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_wrapped_cubelets)


    #!!!Ask for the wrapping shift_and_wrap(1, [0], rest_freq_HI, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, PSF_cubelet)

    return stacked_cube"""

"""def noise_stack_shift(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq):
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
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""
    
"""def noise_stack_random(num_galaxies, num_pixels_X, num_pixels_Y, num_channels_cubelets, num_pixels_cubelets, X_AR_ini, Y_DEC_ini, pixel_X_to_AR, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq):
    
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
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""

"""def noise_stack_Healy(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, show_verifications):
    
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
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
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
    shifted_noise_cubelets = shift_and_wrap(num_galaxies, redshifts, rest_freq_HI, freq_ini, channel_to_freq, num_channels_cubelets, num_pixels_cubelets, noise_cubelets)

    #* Make the stacking of the cubelets
    stacked_noise_cube = stacking_process(num_channels_cubelets, num_pixels_cubelets, shifted_noise_cubelets)

    return stacked_noise_cube"""

def extract_spectrum_from_spatial_circular_region(datacube, wcs, num_channels, center_x, center_y, radius):
    """
    Function that extract the integrated spectrum of a circular region of a datacube.

    • Input
    - datacube [array - float]: Array of fluxes of the datacube
    - wcs [WCS]: WCS of the file
    - num_channels [int]: Number of channels in spectral axis
    - center_x [float]: Horizontal position of the center of the circular region
    - center_y [float]: Vertical position of the center of the circular region
    - radius [float]: radius of the circular region

    • Output
    - integrated_spectrum [array - float]: Spatially integrated fluxes of the datacube
    """

    #* We use a circular aperture
    center_xy = [center_x, center_y]
    aperture = CircularAperture(center_xy, r=radius)

    """plt.imshow(datacube[central_channel], cmap='gray_r', origin='lower')
    aperture.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()"""

    integrated_spectrum = np.zeros(num_channels)
    for i in range(num_channels):
        phot_table = aperture_photometry(datacube[i], aperture, wcs=wcs, method='exact')
        integrated_spectrum[i] = phot_table['aperture_sum'][0]

    return integrated_spectrum

def fit_continuum_of_spectrum(spectrum, num_channels, emission_channel, semirange):
    """
    Function that fits the continuum of a spectrum where lies an emission line.

    • Input
    - spectrum [array - float]: fluxes of the spectrum
    - num_channels [int]: Number of channels in spectral axis
    - emission_channel [int]: Central channel of the emission line
    - semirange [int]: Half of the width of the emission line. Every flux outside the range will be considered as continuum

    • Output
    - new_continuum [array - float]: Continuum fitted and adjusted
    - new_spectrum [array - float]: whole spectrum adjusted
    - new_spectrum [array - float]: emission region adjusted (new_spectrum without new_continuum)
    - mask [array - bool]: 'True' where the spectrum is finite
    """
    
    spectrum_central_region = spectrum[emission_channel-semirange:emission_channel+semirange+1]
    continuum = np.concatenate((spectrum[:emission_channel-semirange-1], np.repeat(np.nan, int(2*semirange+1)), spectrum[emission_channel+semirange:]))
    full_channels = np.linspace(0, num_channels, num_channels)
            
    #?print(len(spectrum[:emission_channel-C-1]), len(spectrum[emission_channel+C:]), num_channels)
    #?print(spectrum[:emission_channel-C])

    #* Let's calculate the standard deviation of the continuum outside the central region
    #!!! First we have to fit it to a 1-degree polynomial so it
    #?First degree fit
    with warnings.catch_warnings(): #?Ignore warnings
        warnings.simplefilter('ignore')
        linfitter = fitting.LinearLSQFitter()
        poly_cont_1 = linfitter(models.Polynomial1D(1), np.linspace(1, num_channels, num_channels)[np.isfinite(continuum)], continuum[np.isfinite(continuum)])

    with warnings.catch_warnings(): #?Ignore warnings
        warnings.simplefilter('ignore')
        # Filter non-finite values from data
        mask = np.isfinite(continuum)
        fitter = fitting.LevMarLSQFitter()
        fitted_model = fitter(poly_cont_1, np.linspace(1, num_channels, num_channels)[mask], continuum[mask])

    fitted_continuum = continuum - fitted_model(np.linspace(1, num_channels, num_channels))

    #* Now we calculate the S/N in the central region, using the sum of the emission and the std_dev of the continuum
    fitted_spectrum = spectrum - fitted_model(full_channels)
    fitted_central_region = spectrum_central_region - fitted_model(np.linspace(emission_channel-semirange, emission_channel+semirange, 2*semirange+1))

    return fitted_continuum, fitted_spectrum, fitted_central_region, mask

def S_N_calculation(datacube, wcs, num_channels, center_x, center_y, emission_channel, L, C):
    """
    Metafunction that calculates the S/N ratio of a datacube in a circular region integrating the spaxels first.

    • Input
    - datacube [array - float]: Array of fluxes of the datacube
    - wcs [WCS]: WCS of the file
    - num_channels [int]: Number of channels in spectral axis
    - center_x [float]: Horizontal position of the center of the circular region
    - center_y [float]: Vertical position of the center of the circular region
    - emission_channel [int]: Central channel of the emission line
    - L [int]: Semirange of pixels of the circular region
    - C [int]: Semirange of channels where lies the emission line

    • Output
    - S_N [float]: S/N ratio of the datacube
    """

    spectrum = extract_spectrum_from_spatial_circular_region(datacube, wcs, num_channels, center_x, center_y, L)

    new_continuum, new_spectrum, new_central_region, mask = fit_continuum_of_spectrum(spectrum, num_channels, emission_channel, C)

    std_dev = np.nanstd(new_continuum)
    S_N = np.nansum(new_central_region)/(std_dev*np.sqrt(2*C+1)) #!!! Should I use abs()?? Some values of the flux are negative

    return S_N

def S_N_measurement_test(datacube, num_pixels_cubelets, num_channels_cubelets, wcs, center_x, center_y, emission_channel, rest_freq_HI, channel_to_freq, flux_units):
    """
    Metafunction that calculates the best combination of (L, C) in order to estimate the S/N ratio of a datacube in a circular region integrating the spaxels first.

    • Input    
    - datacube [array - float]: Array of fluxes of the datacube
    - num_pixels_cubelets [int]: Number of pixels of each cubelet in axes X and Y
    - num_channels_cubelets [int]: Number of channels of each cubelets
    - wcs [WCS]: WCS of the file
    - center_x [float]: Horizontal position of the center of the circular region
    - center_y [float]: Vertical position of the center of the circular region
    - emission_channel [int]: Central channel of the emission line
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - channel_to_freq [float]: Ratio between channel and frequency
    - flux_units [string]: Units of the flux

    • Output
    - L_best [int]: Best semirange of pixels of the circular region
    - C_best [int]: Best semirange of channels where lies the emission line
    - S_N_best [float]: Greater S/N ratio of the datacube calculated with (L, C) = (L_best, C_best)
    """
    
    C_min, C_max = 1, int(num_channels_cubelets/2)-1
    signal_to_noise_ratios = np.zeros((num_pixels_cubelets, C_max-1))
    index_array_L = 0
    S_N_best = -1
    for L in range(1, num_pixels_cubelets+1):
        index_array_C = 0
        
        integrated_spectrum = extract_spectrum_from_spatial_circular_region(datacube, wcs, num_channels_cubelets, center_x, center_y, L)

        for C in range(C_min, C_max):
            new_continuum, new_spectrum, new_central_region, mask = fit_continuum_of_spectrum(integrated_spectrum, num_channels_cubelets, emission_channel, C)

            std_dev = np.nanstd(new_continuum)
            S_N = np.nansum(new_central_region)/(std_dev*np.sqrt(2*C+1)) #!!! Should I use abs()?? Some values of the flux are negative

            if(S_N > S_N_best):
                L_best, C_best = L, C
                S_N_best = S_N
            signal_to_noise_ratios[index_array_L, index_array_C] = np.nansum(new_central_region)/(std_dev*np.sqrt(2*C+1))
            
            #print(Signal_to_noise_ratio)

            
            if (C==50 and L==7):
                full_channels = np.linspace(0, num_channels_cubelets, num_channels_cubelets)
                spectrum_central_region = integrated_spectrum[emission_channel-C:emission_channel+C+1]
                continuum = np.concatenate((integrated_spectrum[:emission_channel-C-1], np.repeat(np.nan, int(2*C+1)), integrated_spectrum[emission_channel+C:]))
    
                fig, ax = plt.subplots(figsize=(12.8,7.20))
                plt.bar(full_channels, integrated_spectrum, color='#264653', label="Original spectrum", bottom=np.nanmax(new_spectrum)*1.5)
                plt.bar(full_channels[mask], continuum[mask], color='#D62828', label="Fitted continuum", alpha=1, bottom=np.nanmax(new_spectrum)*1.5)
                plt.bar(full_channels, new_spectrum, color="#EBC033", label="Corrected continuum")

                #?Figure's labels
                ax.set_xlabel("Wavelength ($\mathrm{\AA}$)", fontsize=16)
                ax.set_ylabel("Flux density (arbitrary units)", fontsize=16)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_xlim(full_channels[0], full_channels[-1])

                #?Save the figure
                ax.legend(loc='best', fontsize=16)
                fig.tight_layout()
                plt.savefig("Verification_process/SN_best/SN_continuum_fit.pdf")

                #!!! C==119 and L==2

                """print(np.linspace(1, center_channel-C-1, center_channel-C-1), len(np.linspace(1, center_channel-C-1, center_channel-C-1)))
                print(np.linspace(center_channel-C, center_channel+C, 2*C+1), len(np.linspace(center_channel-C, center_channel+C, 2*C+1)))
                print(np.linspace(center_channel+C+1, num_channels_cubelets, num_channels_cubelets - center_channel - C), len(np.linspace(center_channel+C+1, num_channels_cubelets, num_channels_cubelets - center_channel - C)))"""

                fig, ax = plt.subplots(figsize=(19.2, 10.8))

                ax.set_xlabel("Channels")
                ax.set_ylabel(r"Flux density (%s)" %flux_units, labelpad=12.0)

                freq_axis = np.linspace(rest_freq_HI-channel_to_freq*num_channels_cubelets/2, rest_freq_HI+channel_to_freq*num_channels_cubelets/2, num_channels_cubelets)*10**(-6)
                chann_axis = np.linspace(1, num_channels_cubelets, num_channels_cubelets)

                ax.grid(True, alpha=0.5, which="minor", ls=":")
                ax.grid(True, alpha=0.7, which="major")
                
                plt.bar(np.linspace(1, emission_channel-C-1, emission_channel-C-1), new_spectrum[:emission_channel-C-1])
                plt.bar(np.linspace(emission_channel-C, emission_channel+C, 2*C+1), new_central_region)
                plt.bar(np.linspace(emission_channel+C+1, num_channels_cubelets, num_channels_cubelets - emission_channel - C), new_spectrum[emission_channel+C:])
                
                ax.vlines(emission_channel, min(spectrum_central_region), max(spectrum_central_region), linestyles='dashdot', color='red', label='HI position', alpha=1, zorder=0)
                
                #plt.plot(full_channels, gauss_function(full_channels, amplitud, media, sigma))

                ax.legend(loc='best')

                plt.tight_layout()

                plt.savefig('Verification_process/SN_best/regions_SN_spectrum_%i_%i.pdf' %(L, C))

                """print(np.nanstd(continuum), np.nanstd(new_continuum))
                print(amplitud)
                print(signal_to_noise_ratios[index_array_L, index_array_C], "\n")
                print(result.fit_report())"""
            index_array_C += 1
        index_array_L += 1
    
    print("Best combination of (L, C) in order to calculate S/N: L=%i, C=%i. Best S/N: %f.\n" %(L_best, C_best, S_N_best))
            
    fig, ax = plt.subplots(figsize=(9, 7.20))
    
    centers = [2, int(num_channels_cubelets/2), 1, num_pixels_cubelets]
    dx, = np.diff(centers[:2])/(signal_to_noise_ratios.shape[1]-1)
    dy, = -np.diff(centers[2:])/(signal_to_noise_ratios.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

    signals = ax.imshow(signal_to_noise_ratios, cmap="plasma", origin='lower', extent=extent, interpolation='nearest', aspect='auto', vmin=0)#, norm=LogNorm())

    plt.colorbar(signals)

    #?Figure's labels
    ax.set_xlabel("Value of C")
    ax.set_ylabel("Value of L")
    ax.set_title("S/N estimation in function of (C, L)", fontsize=20)

    plt.tight_layout()

    plt.savefig("Verification_process/SN_best/signal_to_noise.pdf")

    return L_best, C_best, S_N_best

def plot_spaxel_spectrum(datacube, num_galaxies, rest_freq_HI, channel_to_freq, num_channels_cubelets, flux_units, spaxel_x, spaxel_y, factor, name): #!!! Can I use 'kwargs' for variables like 'num_galaxies', 'factor', 'name'? If yes, how?
    """
    Function that plot the spectrum of the spaxel of a datacube.

    • Input
    - datacube [array - float]: Array of fluxes of the datacube
    - num_galaxies [int]: Number of galaxies of the sample
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - channel_to_freq [float]: Ratio between channel and frequency
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - flux_units [string]: Units of the flux
    - spaxel_x [int]: Horizontal position of the spaxel we are interested in
    - spaxel_y [int]: Vertical position of the spaxel we are interested in
    - factor [float]: rescaling factor of the flux
    - name [string]: name of the final representation

    • Output
    - None
    """
    spectrum = datacube[:, spaxel_y, spaxel_x]*factor #* We extract the spectrum of the spaxel and rescale it

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax2 = ax.twiny()

    ax.set_xlabel("Frequency (MHz)")
    ax2.set_xlabel("Channels")
    ax.set_ylabel(r"Flux density ($10^{-6}$ %s)" %flux_units, labelpad=12.0)

    freq_axis = np.linspace(rest_freq_HI-channel_to_freq*num_channels_cubelets/2, rest_freq_HI+channel_to_freq*num_channels_cubelets/2, num_channels_cubelets)*10**(-6)
    chann_axis = np.linspace(1, num_channels_cubelets, num_channels_cubelets)

    ax.grid(True, alpha=0.5, which="minor", ls=":")
    ax.grid(True, alpha=0.7, which="major")

    ax.plot([], [], label='Number of cubelets stacked: %i' %num_galaxies, alpha=0)

    ax.bar(freq_axis, spectrum, color='black', alpha=1, label='Stacked spectrum', zorder=1, width=0.1)
    ax2.bar(chann_axis, spectrum, visible=False)

    ax.set_ylim(min(spectrum)-1, max(spectrum)+1)

    #for i in range(num_galaxies):
        #plt.plot(new_X_axis, new_spectra[i])

    #plt.plot(new_X_axis, avg_spectrum, '-.', color='midnightblue', alpha=1, label='Average spectrum')

    ax.vlines(rest_freq_HI/1e6, min(spectrum), max(spectrum), linestyles='dashdot', color='red', label='HI position',  alpha=1, zorder=0)

    ax.legend(loc='best')

    plt.tight_layout()

    plt.savefig('%s.pdf' %name)