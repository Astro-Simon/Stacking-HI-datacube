# coding=utf-8

#! Libraries
#from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np

cosmo = FlatLambdaCDM(H0=70*u.km / u.s / u.Mpc, Tcmb0=2.725*u.K, Om0=0.3)

from modules.data_info_extraction_functions import param_file, data_extraction, get_galaxies_positions, cubelets_limits
from modules.stacking_functions import datacube_stack
from modules.S_N_functions import S_N_measurement_test, S_N_calculation
from modules.results import results_catalogue, save_datacube, plot_spaxel_spectrum, calculate_HI_mass

general_path, name_orig_data_cube, name_orig_PSF_cube, name_catalog, column_RA, column_Dec, column_z, path_results, weights_option, degree_fit_continuum, number_of_repetitions, semi_distance_around_galaxies, semi_vel_around_galaxies = param_file('param_file.txt')

# * Number of channels around which the emission is supposed to be located. We use it to extract the continuum of the spectra and calculate sigmas (for weights) !!!Correct value?
central_width = 5  # !!!Also have to rescale it

#! Main code
def main():
    #!!! Use parser to ask for arguments to the user (files to use, num_pixels, num_channels, z_min, z_max, etc.)

    wcs, freq_ref, pixel_X_to_AR, pixel_Y_to_Dec, pixel_scale, channel_to_freq, X_AR_ini, Y_DEC_ini, freq_ini, flux_units, data, min_redshift, max_redshift = data_extraction(name_orig_data_cube, 0)

    #!Extract from the catalog the 3D positions of the galaxies
    coords_RA, coords_DEC, redshifts, num_galaxies = get_galaxies_positions(name_catalog, min_redshift, max_redshift, column_RA, column_Dec, column_z)

    num_pixels_cubelets, num_channels_cubelets, num_pixels_cubelets_final, num_channels_cubelets_final, luminosity_distances, zmin, zmax, central_spaxel, central_channel = cubelets_limits(num_galaxies, redshifts, semi_distance_around_galaxies, pixel_scale, semi_vel_around_galaxies, data, channel_to_freq, freq_ref)

    print(f'\n\nStacking {num_galaxies} cubelets of ~{semi_distance_around_galaxies*2} x {semi_distance_around_galaxies*2} x {2*semi_vel_around_galaxies:.2f} ({zmin:.3f} < z < {zmax:.3f})...\n')

    print("\nDATA STACKING\n")

    #! Get stacked data datacube
    #tic = time.perf_counter()
    stacked_data_cube, integrated_flux_cubelets, errorbar_min, errorbar_max, noise_evolution = datacube_stack('Data', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, freq_ref, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances)
    #toc = time.perf_counter()
    print(f"Data stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'data_stacked_cube', name_orig_data_cube,
                  stacked_data_cube, num_channels_cubelets_final, freq_ref)

    """print("\nPSF STACKING\n")

    #! Get stacked PSF datacube
    PSF = fits.getdata(name_orig_PSF_cube, ext=0)
    PSF = PSF[0]
    stacked_PSF_cube, _, _, _, _ = datacube_stack('PSF', num_galaxies, num_channels_cubelets, 2*num_pixels_cubelets, None, None, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, PSF, wcs, flux_units, redshifts, freq_ref, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances)
    #!!! Should we use weights for the PSF in the stacking process??

    print("PSF stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'PSF_stacked_cube', name_orig_PSF_cube, stacked_PSF_cube, num_channels_cubelets_final, freq_ref)"""

    print("\nNOISE STACKING")

    #! Get stacked mean noise datacube
    # ? Positions switched
    central_spectra = np.zeros((number_of_repetitions, 2*num_channels_cubelets_final+1))
    average_stacked_noise_cube = np.zeros((2*num_channels_cubelets_final + 1, 2*num_pixels_cubelets_final + 1, 2*num_pixels_cubelets_final + 1))
    
    for i in range(number_of_repetitions):
        print(f"\nIteration {i+1}/{number_of_repetitions}")
        stacked_noise_cube, _, _, _, _ = datacube_stack('Noise', num_galaxies, num_channels_cubelets, num_pixels_cubelets, coords_RA, coords_DEC, X_AR_ini, pixel_X_to_AR, Y_DEC_ini, pixel_Y_to_Dec, data, wcs, flux_units, redshifts, freq_ref, freq_ini, channel_to_freq, central_width, central_spaxel, central_channel, weights_option, luminosity_distances)  # !!! Should I re-use the results from the data datacube?

        central_spectra[i] = stacked_noise_cube[:, central_spaxel, central_spaxel]

        average_stacked_noise_cube += stacked_noise_cube

    average_stacked_noise_cube /= number_of_repetitions

    mean_reference_spectrum = np.nanmean(central_spectra, axis=0)
    variance_reference_spectrum = np.nanstd(central_spectra, axis=0)

    print("Noise stacked cube obtained!")

    # Save the datacube
    save_datacube(path_results, 'noise_stacked_cube', name_orig_data_cube, average_stacked_noise_cube, num_channels_cubelets_final, freq_ref)

    #! Calculate SNR
    S_N_data, L_best, C_best = S_N_measurement_test(
        stacked_data_cube, num_pixels_cubelets_final, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, degree_fit_continuum)
    print(f"Best combination of (L, C) in order to calculate S/N: L={L_best}, C={C_best}. Best S/N: {S_N_data:.2f}.\n")

    S_N_noise = S_N_calculation(average_stacked_noise_cube, num_channels_cubelets_final, central_spaxel, central_spaxel, central_channel, L_best, C_best, degree_fit_continuum)
    print(f"S/N of noise cube from switched redshifts: {S_N_noise:.3f}!\n")

    #! Save results
    results_catalogue(path_results, coords_RA, coords_DEC, redshifts, integrated_flux_cubelets)

    spectral_width_SN = C_best * channel_to_freq * c/(1000*u.m/u.s) * (1 + np.max(redshifts)) / freq_ref

    spatial_width_SN = cosmo.angular_diameter_distance(zmax) * 1000 * np.tan(L_best*pixel_scale*np.pi/180) / u.Mpc

    plot_spaxel_spectrum(path_results, stacked_data_cube, mean_reference_spectrum, variance_reference_spectrum, central_spaxel, central_spaxel, errorbar_min, errorbar_max, noise_evolution, freq_ref, channel_to_freq, num_channels_cubelets_final, semi_distance_around_galaxies, semi_vel_around_galaxies, flux_units, num_galaxies, spectral_width_SN, L_best, spatial_width_SN, 6, name='spectrum_plot')
    
    #calculate_HI_mass(stacked_data_cube, central_spaxel, central_spaxel, errorbar_min, errorbar_max, freq_ref, channel_width_zmax, channel_to_freq, num_channels_cubelets_final, redshifts, luminosity_distances, flux_units, num_galaxies)


if __name__ == '__main__':
    main()
