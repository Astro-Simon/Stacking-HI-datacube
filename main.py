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
import matplotlib.pyplot as plt
import numpy as np

from functions import plot_spaxel_spectrum
from data_info_extraction_functions import data_and_catalog_extraction, get_galaxies_positions
from stacking_functions import datacube_stack
from S_N_functions import S_N_measurement_test, S_N_calculation

# warnings.filterwarnings("ignore")

# Define the cosmology used
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

#! Files
general_path = '/home/bonnal/Desktop/JAE'
name_orig_data_cube = 'fullsurvey_1255~1285_image.fits'
name_orig_PSF_cube = 'fullsurvey_1255_1285_psf.fits'
name_catalog = 'G10COSMOSCatv05.csv_z051_sq_chiles_specz'

#! Global parameters
#!!! Use kpc instead of number of pixels and angstroms/Hz instead of number of channels
cosmo = FlatLambdaCDM(H0=70*u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
weights_option = 'fabello'
degree_fit_continuum = 1  #* Degree of fit of continuum around emission lines
show_verifications = False
test = False

#* We are going to extract cubelets of 81x81 kpc^2 around each galaxy for data and noise stack
semi_distance_around_galaxies = 40*u.kpc

#* Number of channels around which the emission is supposed to be located. We use it to extract the continuum of the spectra and calculate sigmas (for weights) !!!Correct value?
central_width = 5 #!!!Also have to rescale it #!!!Also have to rescale it

#* Half-range of velocities around the galaxy emission we select and use in the cubelets
#* Half-range of velocities around the galaxy emission we select and use in the cubelets
semi_vel_around_galaxies = 500 * u.km / u.s

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

    wcs, rest_freq, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, X_AR_final, Y_DEC_ini, Y_DEC_final, freq_ini, freq_final, flux_units, num_pixels_X, num_pixels_Y, num_channels, data, min_redshift, max_redshift = data_and_catalog_extraction(name_orig_data_cube, 0) # !!! Lots of unnecessary values

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

    print(f'\n{mini}. z_min = {zmin}, {num_pixels_cubelets[mini]}, {num_channels_cubelets[mini]}, {num_pixels_cubelets[maxi]/num_pixels_cubelets[mini]}, {num_channels_cubelets[maxi]/num_channels_cubelets[mini]}')
    print(f'{maxi}. z_max = {zmax}, {num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]}, {num_pixels_cubelets[maxi]/num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]/num_channels_cubelets[maxi]}')

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

    print(f'\n{mini}. z_min = {zmin}, {num_pixels_cubelets[mini]}, {num_channels_cubelets[mini]}, {spatial_scales[mini]}, {spectral_scales[mini]}')
    print(f'{maxi}. z_max = {zmax}, {num_pixels_cubelets[maxi]}, {num_channels_cubelets[maxi]}, {spatial_scales[maxi]}, {spectral_scales[maxi]}')

    #* After the stacking we will have a single number of spaxels and channels of the stacked datacube
    num_pixels_cubelets_final = int(num_pixels_cubelets[maxi]) #!!! Use nanmin(num_pixels_cubelets)
    central_spaxel = int(num_pixels_cubelets_final) #* Position of the stacked emission after the stacking #!!! Use nanmin

    num_channels_cubelets_final = int(num_channels_cubelets[maxi]) #!!! Use nanmin(num_channels_cubelets)
    central_channel = int(num_channels_cubelets_final) #* Position of the stacked emission after the stacking #!!! Use nanmin

    if(num_channels_cubelets[maxi] < central_width):
        print("The emission cannot be wider than the width of the whole spectral axis. Modify 'central_width'.\n")
        exit()
    if((2*num_channels_cubelets[maxi]+1) > data.shape[0]):
        print("The cubelets cannot be spectrally wider than the data datacube. Modify 'semi_vel_around_galaxies'.\n")
        exit()
    if((2*num_pixels_cubelets[maxi]+1) > data.shape[1] or (2*num_pixels_cubelets[maxi]+1) > data.shape[2]):
        print("The cubelets cannot be wider spatially than the data datacube. Modify 'semi_distance_around_galaxies'.\n")
        exit()

    print(f'\n\nStacking {num_galaxies} cubelets of ~{semi_distance_around_galaxies*2} x {semi_distance_around_galaxies*2} x {2*semi_vel_around_galaxies:.2f} ({zmin:.3f} < z < {zmax:.3f})...\n')

    print("\nDATA STACKING\n")

    #! Get stacked data datacube
    #tic = time.perf_counter()
    stacked_data_cube = datacube_stack('Data', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, spatial_scales, spectral_scales, weights_option, luminosity_distances, show_verifications)
    #toc = time.perf_counter()
    print(f"Data stacked cube obtained!")

    """#! Calculate best (L, C) combination for S/N measurement
    L_best, C_best, S_N_data = S_N_measurement_test(stacked_data_cube, num_pixels_cubelets_final, num_channels_cubelets_final, wcs, central_spaxel, central_spaxel, central_channel, rest_freq, channel_to_freq, flux_units, degree_fit_continuum)
    print(f"Best combination of (L, C) in order to calculate S/N: L={L_best}, C={C_best}. Best S/N: {S_N_data:.2f}.\n")

    print("\nPSF STACKING\n")

    #! Get stacked PSF datacube
    PSF = fits.getdata(name_orig_PSF_cube, ext=0)
    PSF = PSF[0]
    stacked_PSF_cube = datacube_stack('PSF', num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, spatial_scales, spectral_scales, weights_option, luminosity_distances, show_verifications)
    #!!! Should we use weights for the PSF in the stacking process??

    print("PSF stacked cube obtained!\n")

    print("\nNOISE STACKING\n")

    #! Get stacked noises datacube and calculate their S/N ratio
    # ? Redshifts switched
    if(test):
        sn_values = []
        for i in range(100):
            stacked_noise_cube_Healy = datacube_stack('Noise', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, spatial_scales, spectral_scales, weights_option, luminosity_distances, show_verifications)  # !!! Should I re-use the results from the data datacube?
            print("Healy-noise stacked cube obtained!")

            S_N_noise_Healy = S_N_calculation(stacked_noise_cube_Healy, wcs, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, L_best, C_best, degree_fit_continuum)
            print(f"{i+1}. S/N of noise cube from switched redshifts: {S_N_noise_Healy:.3f}!\n")
            sn_values.append(S_N_noise_Healy)

        print(f"\nResults of the test: <S/N>_{{noise}} = {np.nanmean(sn_values):.4f} +/- {3*np.std(sn_values):.4f}.\n")
    else:
        stacked_noise_cube_Healy = datacube_stack('Noise', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, rest_freq, freq_ini, channel_to_freq, central_width, spatial_scales, spectral_scales, weights_option, luminosity_distances, show_verifications)  # !!! Should I re-use the results from the data datacube?
        print("Healy-noise stacked cube obtained!")

        S_N_noise_Healy = S_N_calculation(stacked_noise_cube_Healy, wcs, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, L_best, C_best, degree_fit_continuum)
        print(f"S/N of noise cube from switched redshifts: {S_N_noise_Healy:.3f}!\n")

    names = ["data_stack.fits", "PSF_stack.fits", "noise_stack_Healy.fits"]
    names_original = [name_orig_data_cube, name_orig_PSF_cube, name_orig_data_cube]
    datacubes = [stacked_data_cube, stacked_PSF_cube]#, stacked_noise_cube_Healy]
    horizontal_dimensions = [2*num_pixels_cubelets_final, 4*num_pixels_cubelets_final, 2*num_pixels_cubelets_final]
    vertical_dimensions = [2*num_pixels_cubelets_final, 4*num_pixels_cubelets_final, 2*num_pixels_cubelets_final]

    #!!! Change the pixel scale. How to go from pc/px to rad/px?

    for name, name_original, cube, dim_x, dim_y in zip(names, names_original, datacubes, horizontal_dimensions, vertical_dimensions):
        #* Now we keep this stacked datacube inside a .fits file
        path = 'Stacked_cubes/'
        if not os.path.isdir(path):
            os.makedirs(path)

        #* We create the new file
        name_stacked_cube = path + name
        fits.writeto(name_stacked_cube, cube, header=fits.open(name_original)[0].header, overwrite=True)  # ?Save the new datacube

        #* We modify the header so it contains the correct information
        # ?Change the value of number of pixels on X axis
        fits.setval(name_stacked_cube, 'CRPIX1', value=dim_x+1)
        # ?Change the value of number of pixels on Y axis
        fits.setval(name_stacked_cube, 'CRPIX2', value=dim_y+1)
        # ?Change the channel of reference: now it's the centered channel
        fits.setval(name_stacked_cube, 'CRPIX3', value=int(num_channels_cubelets_final/2))
        # ?Change the value of the channel of reference: now it's the emission of interest
        fits.setval(name_stacked_cube, 'CRVAL3', value=rest_freq)"""

    #* We plot the spectrum of the central spaxel (where all the galaxies lie)
    plot_spaxel_spectrum(stacked_data_cube, num_galaxies, rest_freq, channel_to_freq, 2*num_channels_cubelets_final+1, flux_units, central_spaxel, central_spaxel, 10**6, 'Results/stacked_data_central_spaxel')


if __name__ == '__main__':
    main()
