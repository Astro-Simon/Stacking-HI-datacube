import pandas as pd
from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SpectralCoord
import matplotlib.pyplot as plt

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


def plot_spaxel_spectrum(path_results, data_datacube, noise_datacube, spaxel_x, spaxel_y, errorbar_min, errorbar_max, rest_freq, channel_to_freq, num_channels_cubelets, flux_units, num_galaxies, factor=1, name='spectrum_plot'):
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
    data_spectrum = data_datacube[:, spaxel_y, spaxel_x] * factor #* We extract the spectrum of the spaxel and rescale it
    noise_spectrum = noise_datacube[:, spaxel_y, spaxel_x] * factor #* We extract the spectrum of the spaxel and rescale it

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax2 = ax.twiny()

    ax.set_xlabel("Relative velocity (km/s)")
    ax2.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(f"Flux density ($10^{{-6}}$ {flux_units})", labelpad=12.0)

    freq_axis = np.linspace(rest_freq - channel_to_freq * num_channels_cubelets,
                             rest_freq + channel_to_freq * num_channels_cubelets, 2*num_channels_cubelets+1) * 1e-6

    vel_axis = SpectralCoord(freq_axis * u.MHz, redshift=0).to(u.km / u.s, doppler_rest=rest_freq * u.Hz, doppler_convention='optical') # type: ignore                                                              

    ax.grid(True, alpha=0.5, which="minor", ls=":")
    ax.grid(True, alpha=0.7, which="major")

    ax.plot([], [], label=f'Number of cubelets stacked: {num_galaxies}', alpha=0)

    yerr = np.stack((errorbar_min, errorbar_max), axis=1).T * factor

    #ax.bar(vel_axis, spectrum, color='black', alpha=1, zorder=1, width=20, label='Stacked spectrum')
    ax.step(x=vel_axis, y=data_spectrum, marker='o', ls='-', color='black', where='mid', zorder=0, label='Stacked spectrum') #!!! Add yerr

    ax.plot(vel_axis, noise_spectrum, marker='o', ls='-', color='gray', zorder=1, label='Reference spectrum')

    ax2.bar(freq_axis, data_spectrum, visible=False)#, label='Stacked spectrum')

    ax.invert_xaxis()
    ax.set_ylim(min(data_spectrum) - 1, max(data_spectrum) + 1)
    
    # set the legend and layout
    ax.legend(loc='best')
    plt.tight_layout()
    
    # save the figure
    plt.savefig(f'{path_results}{name}.pdf')

    from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler
    from specutils import Spectrum1D
    fluxcon = FluxConservingResampler()
    flux = data_spectrum * 10**-6 * u.Unit('Jy') 

    input_spec = Spectrum1D(spectral_axis=freq_axis*u.MHz, flux=flux) 
    print(vel_axis, flux)

    new_disp_grid = np.arange(freq_axis[0], freq_axis[-1], 0.50) * u.MHz
    new_spec_fluxcon = fluxcon(input_spec, new_disp_grid) 

    print(new_spec_fluxcon.spectral_axis, new_spec_fluxcon.flux)
    f, ax = plt.subplots()  
    ax.step(new_spec_fluxcon.spectral_axis, new_spec_fluxcon.flux)  

    plt.show()