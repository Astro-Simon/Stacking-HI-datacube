import os
import warnings
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from photutils.aperture import CircularAperture, aperture_photometry

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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
    
    mask = aperture.to_mask(method='exact')
    
    plt.imshow(mask)
    plt.show()

    plt.imshow(datacube[int(num_channels/2)], cmap='gray_r', origin='lower')
    aperture.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()

    integrated_spectrum = np.zeros(num_channels)
    for i in range(num_channels):
        phot_table = aperture_photometry(datacube[i], aperture, wcs=wcs, method='exact')
        integrated_spectrum[i] = phot_table['aperture_sum'][0]

    return integrated_spectrum

def fit_continuum_of_spectrum(spectrum, num_channels, emission_channel, semirange, degree):
    """
    Function that fits the continuum of a spectrum where lies an emission line.

    • Input
    - spectrum [array - float]: fluxes of the spectrum
    - num_channels [int]: Semirange of channels in spectral axis
    - emission_channel [int]: Central channel of the emission line
    - semirange [int]: Half of the width of the emission line. Every flux outside the range will be considered as continuum

    • Output
    - new_continuum [array - float]: Continuum fitted and adjusted
    - new_spectrum [array - float]: whole spectrum adjusted
    - new_spectrum [array - float]: emission region adjusted (new_spectrum without new_continuum)
    - mask [array - bool]: 'True' where the spectrum is finite
    """

    if(num_channels % 2 == 0):
        full_length = 2*num_channels
    else:
        full_length = 2*num_channels+1


    spectrum_central_region = spectrum[emission_channel-semirange:emission_channel+semirange+1]
    continuum = np.concatenate((spectrum[:emission_channel-semirange], np.repeat(np.nan, int(2*semirange+1)), spectrum[emission_channel+semirange+1:]))
    full_channels = np.linspace(0, full_length, full_length)
            
    #?print(len(spectrum[:emission_channel-C-1]), len(spectrum[emission_channel+C:]), num_channels)
    #?print(spectrum[:emission_channel-C])

    #* Let's calculate the standard deviation of the continuum outside the central region. First we fit the continuum
    #?First degree fit
    with warnings.catch_warnings(): #?Ignaramos los warnings
        warnings.simplefilter('ignore')
        linfitter = fitting.LinearLSQFitter()
        poly_cont_1 = linfitter(models.Polynomial1D(degree), np.linspace(1, full_length, full_length)[np.isfinite(continuum)], continuum[np.isfinite(continuum)])

    with warnings.catch_warnings(): #?Ignaramos los warnings
        warnings.simplefilter('ignore')
        fitter = fitting.LevMarLSQFitter()
        mask = np.isfinite(continuum)
        fitted_model = fitter(poly_cont_1, np.linspace(1, full_length, full_length)[mask], continuum[mask])
        fitted_lines = fitted_model(np.linspace(1, full_length, full_length))

    fitted_continuum = continuum - fitted_lines

    #* Now we calculate the S/N in the central region, using the sum of the emission and the std_dev of the continuum
    fitted_spectrum = spectrum - fitted_model(full_channels)
    fitted_central_region = spectrum_central_region - fitted_model(np.linspace(emission_channel-semirange, emission_channel+semirange, 2*semirange+1))

    return fitted_continuum, fitted_spectrum, fitted_central_region, mask

def plot_spaxel_spectrum(datacube, num_galaxies, rest_freq, channel_to_freq, num_channels_cubelets, flux_units, spaxel_x, spaxel_y, factor, name): #!!! Can I use 'kwargs' for variables like 'num_galaxies', 'factor', 'name'? If yes, how?
    """
    Function that plot the spectrum of the spaxel of a datacube.

    • Input
    - datacube [array - float]: Array of fluxes of the datacube
    - num_galaxies [int]: Number of galaxies of the sample
    - rest_freq [float]: Frequency around which spectra are shifted and wrapped
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

    freq_axis = np.linspace(rest_freq-channel_to_freq*num_channels_cubelets/2, rest_freq+channel_to_freq*num_channels_cubelets/2, num_channels_cubelets)*10**(-6)
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

    ax.vlines(rest_freq/1e6, min(spectrum), max(spectrum), linestyles='dashdot', color='red', label='HI position',  alpha=1, zorder=0)

    ax.legend(loc='best')

    plt.tight_layout()

    plt.savefig('%s.pdf' %name)

"""def gaussian(x, mu, sig, m):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + m*x

x = np.linspace(-3, 3, 120)
semi_size = int(len(x)/2)
y = gaussian(x, 0, 2, 0)

C = 20
center_channel = semi_size+1

new_continuum, new_spectrum, new_central_region, mask = fit_continuum_of_spectrum(y, semi_size, int(semi_size+1), C, 1)
emission_channel = center_channel
full_channels = np.linspace(1, 2*semi_size, 2*semi_size)
spectrum_central_region = y[emission_channel-C:emission_channel+C+1]
continuum = np.concatenate((y[:emission_channel-C-1], np.repeat(np.nan, int(2*C+1)), y[emission_channel+C:]))

fig, ax = plt.subplots(figsize=(12.8, 7.20))
print(len(full_channels), len(y))
plt.bar(full_channels, y, color='#264653', label="Original spectrum", bottom=np.nanmax(new_spectrum)*1.5)
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



fig, ax = plt.subplots(figsize=(19.2, 10.8))

ax.set_xlabel("Channels")
ax.set_ylabel(r"Flux density", labelpad=12.0)

freq_axis = np.linspace(1420-125000*semi_size/2, 1420+125000*semi_size, 2*semi_size+1)*10**(-6)
chann_axis = np.linspace(1, 2*semi_size+1, 2*semi_size+1)

ax.grid(True, alpha=0.5, which="minor", ls=":")
ax.grid(True, alpha=0.7, which="major")

plt.bar(np.linspace(1, emission_channel-C-1, emission_channel-C-1), new_spectrum[:emission_channel-C-1])
plt.bar(np.linspace(emission_channel-C, emission_channel+C, 2*C+1), new_central_region)
print(len(np.linspace(emission_channel+C+1, 2*semi_size+1, 2*semi_size+1 - emission_channel - C)), len(new_spectrum[emission_channel+C:]))
#plt.bar(np.linspace(emission_channel+C+1, 2*semi_size+1, 2*semi_size+1 - emission_channel - C), new_spectrum[emission_channel+C:])

ax.vlines(emission_channel, min(spectrum_central_region), max(spectrum_central_region), linestyles='dashdot', color='red', label='HI position', alpha=1, zorder=0)

#plt.plot(full_channels, gauss_function(full_channels, amplitud, media, sigma))

ax.legend(loc='best')

plt.tight_layout()

plt.savefig(f'Verification_process/SN_best/regions_SN_spectrum_test_{C}.pdf')"""