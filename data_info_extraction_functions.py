import numpy as np
import os
import csv
from astropy.io import fits
from astropy.wcs import WCS
import argparse

def copy_header(name_orig_cube):
    """ 
    Function that extract information from the header of the datacube,
    show it and write it in a file.

    • Input
    - name_orig_cube [string]: name of the datacube with '.fits' extension

    • Output
    - wcs [WCS]: WCS of the file
    - rest_freq [float]: Rest frequency of HI emission Original rest frequency of the datacube
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
        f = open(f"{path}header.txt", "w")
        f.write(repr(hdr))
        f.close()
    except:
        print("\nThe header was not copied to a file.\n")

    #* Parameters obtained from the header
    pixel_X_to_AR = float(hdr['CDELT1'])
    pixel_Y_to_Dec = float(hdr['CDELT2'])
    channel_to_freq = float(hdr['CDELT3'])
    try:
        rest_freq = float(hdr['RESTFRQ'])
    except:
        print("\nWARNING: The rest frequency is missing.")
        exit()

    X_AR_ini = float(hdr['CRVAL1']) - pixel_X_to_AR*float(hdr['CRPIX1'])
    X_AR_final = float(hdr['CRVAL1']) + pixel_X_to_AR*float(hdr['NAXIS1']-1)
    Y_DEC_ini = float(hdr['CRVAL2']) - pixel_Y_to_Dec*float(hdr['CRPIX2'])
    Y_DEC_final = Y_DEC_ini + pixel_Y_to_Dec*float(hdr['NAXIS2']-1)
    freq_ini = float(hdr['CRVAL3']) - channel_to_freq*(float(hdr['CRPIX3']) - 1)
    freq_final = float(hdr['CRVAL3']) - channel_to_freq*float(hdr['CRPIX3']) + channel_to_freq*float(hdr['NAXIS3'])

    flux_units = hdr['BUNIT']

    num_pixels_X = int(hdr['NAXIS1'])
    num_pixels_Y = int(hdr['NAXIS2'])
    num_channels = int(hdr['NAXIS3'])

    pixel_scale = np.sqrt(np.abs(pixel_X_to_AR*pixel_Y_to_Dec)) #!!! Supposing the pixels are squared

    #* Show the results on screen
    print(f"\nNumber of channels: {num_channels}")
    print(f"Units of flux: {hdr['BUNIT']}")
    print(f"Coordinates of first pixel: (AR, Dec) = ({int(X_AR_ini/15)} h {int((X_AR_ini/15-int(X_AR_ini/15))*60)} min {((X_AR_ini/15-int(X_AR_ini/15))*60-int((X_AR_ini/15-int(X_AR_ini/15))*60))*60:.2f} sec, {int(Y_DEC_ini)}º {int((Y_DEC_ini-int(Y_DEC_ini))*60)}' {((Y_DEC_ini-int(Y_DEC_ini))*60-int((Y_DEC_ini-int(Y_DEC_ini))*60))*60:.2f}'')")
    print(f"Coordinates of first pixel: ({X_AR_ini:.2f}º, {Y_DEC_ini:.2f}º)")
    print(f"Coordinates of last pixel: ({X_AR_final:.2f}º, {Y_DEC_final:.2f}º)")
    print(f"Initial frequency: {freq_ini/1e6:.2f} MHz")
    print(f"Final frequency: {freq_final/1e6} MHz")
    print(f"Ratio pixel/right ascension: 1 px = {pixel_X_to_AR:.2e}º")
    print(f"Ratio pixel/declination: 1 px = {pixel_Y_to_Dec:.2e}º")
    print(f"Ratio channel/frequency: 1 channel = {channel_to_freq/1e3:.2f} kHz")

    #* We close the header
    hdul.close()

    #!!! What happens if one of these variables are not in the header? Should I create the variable, try to update it if the value is found, and then return it?
    return wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels

def data_and_catalog_extraction(name_orig_cube, extension):
    """
    Metafunction that extract the array of fluxes of the datacube and information of the header.

    • Input
    - name_orig_cube [string]: name of the datacube with '.fits' extension
    - extension [int]: extension of the datacube where fluxed are stocked

    • Output
    - wcs [WCS]: WCS of the file
    - rest_freq [float]: Rest frequency of HI emission Original rest frequency of the datacube
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
    
    wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels = copy_header(name_orig_cube)

    data = fits.getdata(name_orig_cube, ext=extension)
    data = data[0]

    #* First we determine which redshifts we have access to in this datacube 
    z_min = rest_freq/freq_final - 1
    z_max = rest_freq/freq_ini - 1

    return wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels, data, z_min, z_max

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
