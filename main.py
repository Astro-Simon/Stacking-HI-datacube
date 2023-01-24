# coding=utf-8

#! Program that realize a stacking of the datacube 'fullsurvey_1255~1285_image.fits'.

#todo In order to do that we have access to the optical positions and spectroscopic redshiftss of the galaxies contained in this datacube with the file 'G10COSMOSCatv05.csv_z051_sq_chiles_specz'.

#? The process goes as follow:
#? 1. We read the datacube and get the spectrum of each galaxy using their spatial coordinates (for now we suppose there is only one spectrum per galaxy).
#? 2. We stack this sample of spectra (at first we don't make any separation):
#?   2.1. Put every spectrum at rest frame.
#?   2.2. Make an average sum of the spectra.
#?   2.3. The stacked spectrum contains the HI line emission information.
#?   2.4. We create the stacked image (learning in progress).

#! Libraries
import os

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import sys
import time

from functions import plot_spaxel_spectrum
from data_info_extraction_functions import data_and_catalog_extraction, get_galaxies_positions
from stacking_functions import datacube_stack
from S_N_functions import S_N_measurement_test, S_N_calculation

#warnings.filterwarnings("ignore")

# Define the cosmology used
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

#! Figure properties !!! Can we put it here or does it have to be in functions.py? Better to make a file that plot things
plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": 'serif',
        "text.usetex": False,
        "figure.subplot.top": 0.9,
        "figure.subplot.right": 0.9,
        "figure.subplot.left": 0.15,
        "figure.subplot.bottom": 0.12,
        "figure.subplot.hspace": 0.4,
        "savefig.dpi": 180,
        "savefig.format": "png",
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "xtick.minor.size": 2.25,
        "xtick.major.pad": 7.5,
        "xtick.minor.pad": 7.5,
        "ytick.major.pad": 7.5,
        "ytick.minor.pad": 7.5,
        "ytick.major.size": 5,
        "ytick.minor.size": 2.25,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "legend.framealpha": 1,
        "figure.titlesize": 20,
        "lines.linewidth": 2,
    }
)

#! Parameters
show_verifications = False
general_path = '/home/bonnal/Desktop/JAE'
name_orig_data_cube = 'fullsurvey_1255~1285_image.fits'
name_orig_PSF_cube = 'fullsurvey_1255_1285_psf.fits'
name_catalog = 'G10COSMOSCatv05.csv_z051_sq_chiles_specz'


#! Main code
wcs, rest_freq_HI, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels, data, z_min, z_max = data_and_catalog_extraction(name_orig_data_cube, 0)

print('\nWe are going to stack galaxies with redshift between %.3f < z < %.3f.\n' %(z_min, z_max))

coords_RA, coords_DEC, redshifts, num_galaxies = get_galaxies_positions(name_catalog, z_min, z_max)

#todo We decide
#!!! Use kpc instead of number of pixels and angstroms/Hz instead of number of channels
num_pixels_cubelets = 10 #*We are going to extract cubelets of 20x20 px^2 around each galaxy for data and noise stack
central_width = 25 #* Number of channels around which the emission is supposed to be located. We use it to extract the continuum of the spectra and calculate sigmas (for weights) !!!Correct value?
num_channels_cubelets = num_channels #*Half-range of channels around the galaxy emission we select and use in the cubelets
central_spaxel = int(num_pixels_cubelets+1)
if(num_channels_cubelets%2==0): # Even number of channels
    emission_channel = int(num_channels_cubelets/2)
else: # Odd number of channels
    emission_channel = int(num_channels_cubelets/2) + 1


print('Stacking %i cubelets of %i"x%i"x%.2f MHz...' %(num_galaxies, int(abs(pixel_X_to_AR)*3600*(num_pixels_cubelets+1)), int(abs(pixel_Y_to_Dec)*3600*(num_pixels_cubelets+1)), num_channels_cubelets*channel_to_freq/1e6))

#! Get stacked data datacube
tic = time.perf_counter()
stacked_data_cube = datacube_stack('Data', num_galaxies, num_channels_cubelets, num_pixels_cubelets, emission_channel, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, central_width, show_verifications)
toc = time.perf_counter()
print(f"\nData stacked cube obtained in {(toc - tic):0.4f} seconds!")

#! Calculate best (L, C) combination for S/N measurement
L_best, C_best, S_N_data = S_N_measurement_test(stacked_data_cube, num_pixels_cubelets, num_channels_cubelets, wcs, central_spaxel, central_spaxel, emission_channel, rest_freq_HI, channel_to_freq, flux_units)

#! Get stacked PSF datacube
PSF = fits.getdata(name_orig_PSF_cube, ext=0)
PSF = PSF[0]
stacked_PSF_cube = datacube_stack('PSF', num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets, emission_channel, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, central_width, show_verifications)
#!!! Should we use weights for the PSF in the stacking process??

print("\nPSF stacked cube obtained!")

#! Get stacked noises datacube and calculate their S/N ratio
"""#? Positions shifted
stacked_noise_cube_shift = noise_stack_shift(num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq)
print("\nShift-noise stacked cube obtained!")

S_N_noise_shift = S_N_calculation(stacked_noise_cube_shift, wcs, num_channels_cubelets, central_spaxel, central_spaxel, emission_channel, L_best, C_best)
print("S/N of noise cube from shifted positions: %f!\n" %S_N_noise_shift)

#? Random positions
stacked_noise_cube_random = noise_stack_random(num_galaxies, num_pixels_X, num_pixels_Y, num_channels_cubelets, num_pixels_cubelets, X_AR_ini, Y_DEC_ini, pixel_X_to_AR, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq)
print("\nRandom-noise stacked cube obtained!")

S_N_noise_random = S_N_calculation(stacked_noise_cube_random, wcs, num_channels_cubelets, central_spaxel, central_spaxel, emission_channel, L_best, C_best)
print("S/N of noise cube from random positions: %f!\n" %S_N_noise_random)"""

#? Redshifts switched
stacked_noise_cube_Healy = datacube_stack('Noise', num_galaxies, num_channels_cubelets, num_pixels_cubelets, emission_channel, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq_HI, freq_ini, channel_to_freq, central_width, show_verifications) #!!! Should I re-use the results from the data datacube?
print("\nHealy-noise stacked cube obtained!")

S_N_noise_Healy = S_N_calculation(stacked_noise_cube_Healy, wcs, num_channels_cubelets, central_spaxel, central_spaxel, emission_channel, L_best, C_best)
print("S/N of noise cube from switched redshifts: %f!\n" %S_N_noise_Healy)

names = ["data_stack.fits", "PSF_stack.fits", "noise_stack_Healy.fits"]
names_original = [name_orig_data_cube, name_orig_PSF_cube, name_orig_data_cube]
datacubes = [stacked_data_cube, stacked_PSF_cube, stacked_noise_cube_Healy]
horizontal_dimensions = [2*num_pixels_cubelets, 4*num_pixels_cubelets, 2*num_pixels_cubelets]
vertical_dimensions = [2*num_pixels_cubelets, 4*num_pixels_cubelets, 2*num_pixels_cubelets]

for name, name_original, cube, dim_x, dim_y in zip(names, names_original, datacubes, horizontal_dimensions, vertical_dimensions):
    #* Now we keep this stacked datacube inside a .fits file
    path = 'Stacked_cubes/'
    if not os.path.isdir(path):
        os.makedirs(path)

    #* We create the new file
    name_stacked_cube = path + name
    fits.writeto(name_stacked_cube, cube, header=fits.open(name_original)[0].header, overwrite=True) #?Save the new datacube

    #* We modify the header so it contains the correct information
    fits.setval(name_stacked_cube, 'CRPIX1', value=dim_x+1) #?Change the value of number of pixels on X axis
    fits.setval(name_stacked_cube, 'CRPIX2', value=dim_y+1) #?Change the value of number of pixels on Y axis
    fits.setval(name_stacked_cube, 'CRPIX3', value=int(num_channels_cubelets/2)) #?Change the channel of reference: now it's the centered channel
    fits.setval(name_stacked_cube, 'CRVAL3', value=rest_freq_HI) #?Change the value of the channel of reference: now it's the HI emission

#* We plot the spectrum of the central spaxel (where all the galaxies lie)
plot_spaxel_spectrum(stacked_data_cube, num_galaxies, rest_freq_HI, channel_to_freq, num_channels_cubelets, flux_units, central_spaxel, central_spaxel, 10**6, 'Results/stacked_data_central_spaxel')