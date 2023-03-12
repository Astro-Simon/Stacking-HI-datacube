# coding=utf-8

#! Program that realize a stacking of the datacube 'fullsurvey_1255~1285_image.fits'.

# In order to do that we have access to the optical positions and spectroscopic redshiftss of the galaxies contained in this datacube with the file 'G10COSMOSCatv05.csv_z051_sq_chiles_specz'.

# ? The process goes as follow:
# ? 1. We read the datacube and get the spectrum of each galaxy using their spatial coordinates (for now we suppose there is only one spectrum per galaxy).
# ? 2. We stack this sample of spectra (at first we don't make any separation):
# ?   2.1. Put every spectrum at rest frame.
# ?   2.2. Make an average sum of the spectra.
# ?   2.3. The stacked spectrum contains the HI line emission information.
# ?   2.4. We create the stacked image (learning in progress).

#! Libraries
import os

#from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import Angle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_info_extraction_functions import param_file, data_and_catalog_extraction, get_galaxies_positions
from stacking_functions import datacube_stack
from S_N_functions import S_N_measurement_test, S_N_calculation
from results import results_catalogue, save_datacube, plot_spaxel_spectrum

general_path, name_orig_data_cube, name_orig_PSF_cube, name_catalog, path_results, weights_option, degree_fit_continuum, bool_calculate_SNR, semi_distance_around_galaxies, semi_vel_around_galaxies = param_file('param_file.txt')

cosmo = FlatLambdaCDM(H0=70*u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

#* Number of channels around which the emission is supposed to be located. We use it to extract the continuum of the spectra and calculate sigmas (for weights) !!!Correct value?
central_width = 5 #!!!Also have to rescale it

#! Main code
def main():
    #!!! Use parser to ask for arguments to the user (files to use, num_pixels, num_channels, z_min, z_max, etc.)
    """parser = ArgumentParser(description="Create cubelets around galaxies from a datacube using a catalog and cubelets and stack them. \n"  "Only works with wcs=True (for now).",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('-c', '--catalog', required=True,
                        help='Required: Specify the input XML or ascii catalog name. No default.')

    parser.add_argument('-id', '--source-id', default=[], nargs='*', type=int,
                        help='Space-separated list of sources to include in the plotting. Default all sources')

    #parser.add_argument('-x', '--suffix', default='png',
    #                    help='Optional: specify the output image file type: png, pdf, eps, jpeg, tiff, etc (default: %(default)s).')

    parser.add_argument('-o', '--original', default=None,
                        help='Optional: specify the original fits data: used for plotting HI spectra *with* noise over \n'
                            ' the full frequency range of the cube. Otherwise, plot with noise over frequency range\n'
                            ' in the cubelet.  Uses 2D mask to integrate. (No default).')

    parser.add_argument('-b', '--beam', default=None,
                        help='Optional: specify the beam dimensions (bmaj,bmin,bpa) in arcsec, arcsec, deg. If only 1 value\n'
                            ' is given, assume a circular beam. If 2 values are given, assume PA = 0. (No default).')

    parser.add_argument('-cw', '--chan_width', default=[None], nargs=1, type=float,
                        help='Optional: specify the channel width in native units of the original data (e.g. Hz or m/s).'
                             ' (No default).')

    parser.add_argument('-i', '--image_size', default=[6], nargs=1, type=float,
                        help='Optional: specify the minimum survey image size to retrieve in arcmin.  It will be adjusted if\n'
                            ' the HI mask is larger. Note max panstarrs image size is 8 arcmin (default: %(default)s).')

    parser.add_argument('-snr', '--snr-range', default=[2., 3.], nargs=2, type=float,
                        help='Optional: specify which SNRmin and SNRmax values should be used to set the lowest reliable HI \n'
                            ' contour in the figures. The contour level is calculated as the median value in the mom0 image\n'
                            ' of all pixels whose SNR value is within the given range. Default is [2,3].')

    parser.add_argument('-s', '--surveys', default=[], nargs='*', type=str,
                        help='Specify SkyView surveys to retrieve from astroquery on which to overlay HI contours.\n'
                            ' These additional non-SkyView options are also available: \'decals\',\'panstarrs\',\'hst\'.\n'
                            ' \'hst\' only refers to COSMOS HST (e.g. for CHILES). Default is "DSS2 Blue" if no user\n' 
                            ' provided image.')

    parser.add_argument('-m', '--imagemagick', nargs='?', type=str, default='', const='convert',
                        help='Optional: combine the main plots into single large image file using the IMAGEMAGICK CONVERT task.\n'
                            ' If this option is given with no argument we simply assume that CONVERT is executed by the "convert"\n'
                            ' command. Otherwise, the argument of this option gives the full path to the CONVERT executable.\n'
                            ' Only the first multiwavelength image specified in "surveys" argument is plotted next to the\n'
                            ' spectral line data.')

    parser.add_argument('-ui', '--user-image', default=None,
                        help='Optional: Full path to the FITS image on which to overlay HI contours.')

    parser.add_argument('-ur', '--user-range', default=[10., 99.], nargs=2, type=float,
                        help='Optional: Percentile range used when displaying the user image (see "-ui"). Default is [10,99].')

    ###################################################################

    # Parse the arguments above
    args = parser.parse_args()
    suffix = args.suffix
    original = args.original
    imagemagick = args.imagemagick"""

    wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, Y_DEC_ini, freq_ini, flux_units, data, min_redshift, max_redshift = data_and_catalog_extraction(name_orig_data_cube, 0) 
    
    freq_to_vel = u.doppler_optical(rest_freq*u.Hz) #!!! Convention used by the user (possible option)
    freq_to_vel = u.doppler_optical(rest_freq*u.Hz) #!!! Convention used by the user (possible option)
    semi_freq_around_galaxies = abs((semi_vel_around_galaxies).to(u.MHz, equivalencies=freq_to_vel) - rest_freq*u.Hz)

    #!Extract from the catalog the 3D positions of the galaxies
    coords_RA, coords_DEC, redshifts, num_galaxies = get_galaxies_positions(name_catalog, min_redshift, max_redshift)

    #* For each galaxy we calculate the number of pixels we need to get the same physical area determined by semi_distance_around_galaxies
    num_pixels_cubelets = np.zeros(num_galaxies)
    num_channels_cubelets = np.zeros(num_galaxies)
    zmin = np.nanmin(redshifts)
    zmax = np.nanmax(redshifts)
    luminosity_distances = []
    mini, maxi = -1, -1
    for index, z in enumerate(redshifts):
        d_A = cosmo.angular_diameter_distance(z)
        luminosity_distances.append((1+z)**2 * d_A)
        theta = Angle(np.arctan(semi_distance_around_galaxies/d_A), u.radian)
        num_pixels_cubelets[index] = np.ceil(theta.degree/pixel_scale)

        if(z == zmin):
            mini = index
        elif(z == zmax):
            maxi = index
        semi_freq_width = semi_freq_around_galaxies * 1/(1+z)
        num_channels_cubelets[index] = int(np.ceil((semi_freq_width/(channel_to_freq*u.Hz)).decompose()))

    #!!! Apply scaling only if there is a difference between z_min and z_max in num_channels and num_pixels

    num_pixels_cubelets = np.array(num_pixels_cubelets, dtype=int)

    num_channels_cubelets = np.array(num_channels_cubelets, dtype=int) #!!! We suppose that the value of semi_freq_around_galaxies is given for z = 0

    """print(f'\n{mini}. z_min = {zmin}, {num_pixels_cubelets[mini]}, {num_channels_cubelets[mini]}, {num_pixels_cubelets[maxi]/num_pixels_cubelets[mini]}, {num_channels_cubelets[maxi]/num_channels_cubelets[mini]}')
    print(f'{maxi}. z_max = {zmax}, {num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]}, {num_pixels_cubelets[maxi]/num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]/num_channels_cubelets[maxi]}')"""

    #* We calculate the spatial and spectral scales necessary for the scaling algorithms
    spatial_scales = np.zeros(num_galaxies)
    spectral_scales = np.zeros(num_galaxies)
    semi_freq_width_z_max = semi_freq_around_galaxies * 1/(1+zmax)
    for index, z in enumerate(redshifts):
        d_A = cosmo.angular_diameter_distance(z)
        theta = Angle(np.arctan(semi_distance_around_galaxies/d_A), u.radian)

        spatial_scales[index] = Angle(np.arctan(semi_distance_around_galaxies/cosmo.angular_diameter_distance(zmax)), u.radian)/(theta)

        semi_freq_width = semi_freq_around_galaxies * 1/(1+z)
        spectral_scales[index] = semi_freq_width_z_max/semi_freq_width

    #print(f'\n{mini}. z_min = {zmin}, {num_pixels_cubelets[mini]}, {num_channels_cubelets[mini]}, {spatial_scales[mini]}, {spectral_scales[mini]}')
    #print(f'{maxi}. z_max = {zmax}, {num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]}, {spatial_scales[maxi]}, {spectral_scales[maxi]}')

    #* After the stacking we will have a single number of spaxels and channels of the stacked datacube
    num_pixels_cubelets_final = int(num_pixels_cubelets[maxi]) #!!! Use nanmin(num_pixels_cubelets)
    central_spaxel = int(num_pixels_cubelets_final) #* Position of the stacked emission after the stacking #!!! Use nanmin

    num_channels_cubelets_final = int(num_channels_cubelets[maxi]) #!!! Use nanmin(num_channels_cubelets)
    central_channel = int(num_channels_cubelets_final) #* Position of the stacked emission after the stacking #!!! Use nanmin

    if(num_channels_cubelets[maxi] < central_width):
        print("The emission cannot be wider than the width of the whole spectral axis. Reduce 'central_width'.\n")
        exit()
    if((2*num_channels_cubelets[maxi]+1) > data.shape[0]):
        print("The cubelets cannot be spectrally wider than the data datacube. Reduce 'semi_vel_around_galaxies'.\n")
        exit()
    if((2*num_pixels_cubelets[maxi]+1) > data.shape[1] or (2*num_pixels_cubelets[maxi]+1) > data.shape[2]):
        print("The cubelets cannot be wider spatially than the data datacube. Reduce 'semi_distance_around_galaxies'.\n")
        exit()

    print(f'\n\nStacking {num_galaxies} cubelets of ~{semi_distance_around_galaxies*2} x {semi_distance_around_galaxies*2} x {2*semi_vel_around_galaxies:.2f} ({zmin:.3f} < z < {zmax:.3f})...\n')

    print("\nDATA STACKING\n")

    #! Get stacked data datacube
    #tic = time.perf_counter()
    stacked_data_cube, integrated_flux_cubelets, errorbar_min, errorbar_max = datacube_stack('Data', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances, show_verifications)
    #toc = time.perf_counter()
    print(f"Data stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'data_stacked_cube', name_orig_data_cube, stacked_data_cube, num_channels_cubelets_final, rest_freq)
    
    """print("\nPSF STACKING\n")

    #! Get stacked PSF datacube
    PSF = fits.getdata(name_orig_PSF_cube, ext=0)
    PSF = PSF[0]
    stacked_PSF_cube, _, _, _ = datacube_stack('PSF', num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances, show_verifications)
    #!!! Should we use weights for the PSF in the stacking process??

    print("PSF stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'PSF_stacked_cube', name_orig_PSF_cube, stacked_PSF_cube, num_channels_cubelets_final, rest_freq)"""

    print("\nNOISE STACKING\n")

    #! Get stacked noises datacube and calculate their S/N ratio
    # ? Redshifts switched

    stacked_noise_cube, _, _, _ = datacube_stack('Noise', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances, show_verifications)  # !!! Should I re-use the results from the data datacube?
    print("Noise stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'noise_stacked_cube', name_orig_data_cube, stacked_noise_cube, num_channels_cubelets_final, rest_freq)


    #! Calculate SNR
    if(bool_calculate_SNR):
        S_N_data, L_best, C_best = S_N_measurement_test(stacked_data_cube, num_pixels_cubelets_final, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, degree_fit_continuum)
        print(f"Best combination of (L, C) in order to calculate S/N: L={L_best}, C={C_best}. Best S/N: {S_N_data:.2f}.\n")

        S_N_noise = S_N_calculation(stacked_noise_cube, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, L_best, C_best, degree_fit_continuum)
        print(f"S/N of noise cube from switched redshifts: {S_N_noise:.3f}!\n")

    #! Save results
    results_catalogue(path_results, coords_RA, coords_DEC, redshifts, integrated_flux_cubelets)
    
    plot_spaxel_spectrum(path_results, stacked_data_cube, stacked_noise_cube, central_spaxel, central_spaxel, errorbar_min, errorbar_max, rest_freq, channel_to_freq, num_channels_cubelets_final, flux_units, num_galaxies, factor=10**6, name='spectrum_plot')

if __name__ == '__main__':
    main()
