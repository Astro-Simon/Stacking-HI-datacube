import pandas as pd
from astropy.io import fits
from astropy.coordinates import SpectralCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from modules.functions import extract_spectrum_from_spatial_circular_region

def results_catalogue(path, coords_RA, coords_DEC, redshifts, integrated_flux_cubelets):
    data = {'Redshift': redshifts,
            'RA': coords_RA,
            'Dec': coords_DEC,
            'Int. flux': integrated_flux_cubelets}
    
    # Create the pandas DataFrame
    df = pd.DataFrame(data)

    # Save it to a CSV
    filename = f'{path}results_catalogue.csv'
    df.to_csv(filename, sep='\t', encoding='utf-8', index=False)

    print(f"\nCatalogue of cubelets created. Check for the file '{filename}'.\n")

def save_datacube(path, name, name_original, datacube, num_channels_cubelets_final, rest_freq):

    horizontal_dimension = 2*datacube.shape[2] + 1
    vertical_dimension = 2*datacube.shape[1] + 1

    #* We create the new file
    filename = f'{path}{name}.fits'
    fits.writeto(filename, datacube, header=fits.open(name_original)[0].header, overwrite=True) # type: ignore    
    
    #* We modify the header so it contains the correct information
    # ?Change the value of number of pixels on X axis
    fits.setval(filename, 'CRPIX1', value=horizontal_dimension)
    # ?Change the value of number of pixels on Y axis
    fits.setval(filename, 'CRPIX2', value=vertical_dimension)
    # ?Change the channel of reference: now it's the centered channel
    fits.setval(filename, 'CRPIX3', value=num_channels_cubelets_final)
    # ?Change the value of the channel of reference: now it's the emission of interest
    fits.setval(filename, 'CRVAL3', value=rest_freq)

def plot_spaxel_spectrum(path_results, data_datacube, mean_reference_spectrum, variance_reference_spectrum, spaxel_x, spaxel_y, errorbar_min, errorbar_max, noise_evolution, rest_freq, channel_to_freq, num_channels_cubelets, semi_distance_around_galaxies, semi_vel_around_galaxies, flux_units, num_galaxies, spectral_width_SN, L_best, spatial_width_SN, exponent=0, name='spectrum_plot'):
    """
    Plot the spectrum of a single spaxel in a datacube.

    Parameters
    ----------
    datacube : `~numpy.ndarray`
        The datacube containing the spectrum.
    spaxel_x : int
        The x-coordinate of the spaxel.
    spaxel_y : int
        The y-coordinate of the spaxel.
    rest_freq : float
        The rest frequency of the spectral line in Hz.
    channel_to_freq : float
        The conversion factor between channel number and frequency in MHz.
    num_channels_cubelets : int
        The number of channels in each cubelet.
    flux_units : str
        The units of the flux density.
    num_galaxies : int, optional
        The number of galaxies to stack, by default 1.
    factor : float, optional
        The factor to scale the spectrum by, by default 1.
    name : str, optional
        The name of the output file, by default 'spectrum_plot.pdf'.

    Returns
    -------
    None

    """
    fig = plt.figure(constrained_layout=False, figsize=(15, 10))

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax3_2 = ax3.twiny()
    
    # ------------------------- First plot ------------------------- #
    x = np.arange(1, num_galaxies+1, 1)
    ax1.plot(x, noise_evolution/x, ls='-', color='b', marker='o', ms=4)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of stacked cubelets')
    ax1.set_ylabel(r'$\sigma$ of stacked datacube')


    # ------------------------- Second plot ------------------------- #
    image2 = ax2.imshow(data_datacube[num_channels_cubelets, :, :]*10**exponent, origin='lower', cmap='plasma', extent=[-semi_distance_around_galaxies/u.kpc, semi_distance_around_galaxies/u.kpc, -semi_distance_around_galaxies/u.kpc, semi_distance_around_galaxies/u.kpc], interpolation='nearest', aspect='equal')
    circ = Circle((0, 0), spatial_width_SN, facecolor='g', alpha=0.25, edgecolor='None')
    ax2.add_patch(circ)
    

    ax2.set_xlabel('x (kpc)')
    ax2.set_ylabel('y (kpc)')

    cbar2 = fig.colorbar(image2, ax=ax2)
    cbar2.ax.set_ylabel(f'Flux density $(10^{exponent} {flux_units})$')
    
    # ------------------------- Third plot ------------------------- #
    data_spectrum = data_datacube[:, spaxel_y, spaxel_x] * 10**exponent
    noise_spectrum = mean_reference_spectrum * 10**exponent
    variance_reference_spectrum *= 10**exponent
    ax3.set_xlabel("Relative velocity (km/s)")
    ax3_2.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel(f"Flux $(10^{exponent} {flux_units})$", labelpad=12.0)

    freq_axis = np.linspace(rest_freq - channel_to_freq * num_channels_cubelets,
                             rest_freq + channel_to_freq * num_channels_cubelets, 2*num_channels_cubelets+1) * 1e-6

    vel_axis = SpectralCoord(freq_axis * u.MHz, redshift=0).to(u.km / u.s, doppler_rest=rest_freq * u.Hz, doppler_convention='optical') # type: ignore                                                              

    ax3.grid(True, alpha=0.5, which="minor", ls=":")
    ax3.grid(True, alpha=0.7, which="major")

    ax3.plot([], [], label=f'Number of cubelets stacked: {num_galaxies}', alpha=0)

    yerr = np.stack((errorbar_min, errorbar_max), axis=1).T

    #print(yerr)

    ax3.plot(vel_axis, noise_spectrum, marker='o', ls='-', color='gray', zorder=1, label='Average stacked reference spectrum')

    ax3.fill_between(vel_axis/(u.km/u.s), noise_spectrum-variance_reference_spectrum, noise_spectrum+variance_reference_spectrum, color='orange', alpha=0.25, zorder=0, label=f'1-$\sigma$ variance')

    #ax.bar(vel_axis, spectrum, color='black', alpha=1, zorder=1, width=20, label='Stacked spectrum')
    ax3.errorbar(x=vel_axis, y=data_spectrum, yerr=yerr, marker='o', ls='', color='black', zorder=2, label='Stacked spectrum') #!!! Add yerr
    ax3.step(x=vel_axis, y=data_spectrum, marker='o', ls='-', color='black', where='mid', zorder=2) #!!! Add yerr

    ax3.axvspan(-spectral_width_SN, spectral_width_SN, alpha=0.15, color='green', zorder=-1)

    ax3_2.bar(freq_axis, data_spectrum, visible=False)#, label='Stacked spectrum')

    ax3.invert_xaxis()
    #ax3.set_ylim(min(data_spectrum) - 1, max(data_spectrum) + 1)
    
    # set the legend and layout
    ax3.legend(loc='best')
    plt.tight_layout()
    
    # save the figure
    plt.savefig(f'{path_results}{name}.pdf')
    plt.savefig(f'{path_results}{name}.png')

    """from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler
    from specutils import Spectrum1D
    fluxcon = FluxConservingResampler()
    flux = data_spectrum * 10**-6 * u.Unit('Jy') 

    input_spec = Spectrum1D(spectral_axis=freq_axis*u.MHz, flux=flux) 
    print(vel_axis, flux)

    new_disp_grid = np.arange(freq_axis[0], freq_axis[-1], 0.50) * u.MHz
    new_spec_fluxcon = fluxcon(input_spec, new_disp_grid) 

    print(new_spec_fluxcon.spectral_axis, new_spec_fluxcon.flux)
    f, ax = plt.subplots()  
    ax.step(new_spec_fluxcon.spectral_axis, new_spec_fluxcon.flux)"""


def calculate_HI_mass(data_datacube, spaxel_x, spaxel_y, errorbar_min, errorbar_max, rest_freq, channel_to_freq, channel_width_zmax, num_channels_cubelets, redshifts, luminosity_distances, flux_units, num_galaxies):

    data_spectrum = data_datacube[:, spaxel_y, spaxel_x] #* We extract the spectrum of the spaxel and rescale it
    
    yerr = np.stack((errorbar_min, errorbar_max), axis=1).T

    integrated_flux = np.nansum(data_spectrum) #!!! Have to integrate over a window

    HI_mass = (2.36*10**5/(1+np.nanmean(redshifts))) * np.nanmean(np.square(luminosity_distances)) * integrated_flux * channel_width_zmax

    integrated_flux_err = np.nanmean([np.nansum(np.abs(errorbar_min)), np.nansum(np.abs(errorbar_max))])

    HI_mass_err = (2.36*10**5/(1+np.nanmean(redshifts))) * np.nanmean(np.square(luminosity_distances)) * integrated_flux_err * channel_width_zmax

    print(f"Total HI mass estimation: {HI_mass:.2e} +\- {HI_mass_err:.2e} solar masses.\n")