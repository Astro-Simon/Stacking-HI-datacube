from functions import extract_spectrum_from_spatial_circular_region, fit_continuum_of_spectrum
import numpy as np
import matplotlib.pyplot as plt

def S_N_calculation(datacube, wcs, num_channels, center_x, center_y, emission_channel, L, C, degree_fit_continuum):
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

    spectrum = extract_spectrum_from_spatial_circular_region(datacube, wcs, 2*num_channels+1, center_x, center_y, L)

    x_axis = np.arange(2*num_channels+1)
    new_spectrum, new_central_region, fitted_continuum = fit_continuum_of_spectrum(spectrum, x_axis, emission_channel, C, degree_fit_continuum)

    std_dev = np.nanstd(new_central_region)
    S_N = np.nansum(new_central_region)/(std_dev*np.sqrt(2*C+1))

    return S_N

def S_N_measurement_test(datacube, num_pixels_cubelets, num_channels_cubelets, wcs, center_x, center_y, emission_channel, rest_freq, channel_to_freq, flux_units, degree_fit_continuum):
    """
    Metafunction that calculates the best combination of (L, C) in order to estimate the S/N ratio of a datacube in a circular region integrating the spaxels first.

    • Input    
    - datacube [array - float]: Array of fluxes of the datacube
    - num_pixels_cubelets [int]: Semirange of pixels of each cubelet in axes X and Y
    - num_channels_cubelets [int]: Semirange of channels of each cubelets
    - wcs [WCS]: WCS of the file
    - center_x [float]: Horizontal position of the center of the circular region
    - center_y [float]: Vertical position of the center of the circular region
    - emission_channel [int]: Central channel of the emission line
    - rest_freq [float]: Frequency around which spectra are shifted and wrapped
    - channel_to_freq [float]: Ratio between channel and frequency
    - flux_units [string]: Units of the flux

    • Output
    - L_best [int]: Best semirange of pixels of the circular region
    - C_best [int]: Best semirange of channels where lies the emission line
    - S_N_best [float]: Greater S/N ratio of the datacube calculated with (L, C) = (L_best, C_best)
    """

    S_N_best = -1
    L_best, C_best = 0, 0
    signal_to_noise_ratios = np.zeros((num_pixels_cubelets, num_channels_cubelets))
    center_channel = num_channels_cubelets+1
    index_array_L = 0
    for L in range(1, num_pixels_cubelets+1):
        integrated_spectrum = extract_spectrum_from_spatial_circular_region(datacube, wcs, 2*num_channels_cubelets+1, center_x, center_y, L)
        """print(len(integrated_spectrum))
        plt.bar(np.linspace(0, len(integrated_spectrum), len(integrated_spectrum)), integrated_spectrum)
        plt.axvline(17.5)
        plt.show()"""
        index_array_C = 0
        for C in range(1, num_channels_cubelets+1):
            x_axis = np.arange(2*num_channels_cubelets+1)
            new_spectrum, new_central_region, fitted_continuum = fit_continuum_of_spectrum(integrated_spectrum, x_axis, emission_channel, C, degree_fit_continuum)

            std_dev = np.nanstd(new_central_region)

            S_N = np.nansum(new_central_region)/(std_dev*np.sqrt(2*C+1))

            if S_N >= S_N_best:
                L_best, C_best = L, C
                S_N_best = S_N
            signal_to_noise_ratios[index_array_L, index_array_C] = S_N

            if (C==13 and L==3):
                full_channels = np.linspace(0, 2*num_channels_cubelets+1, 2*num_channels_cubelets+1)
                spectrum_central_region = integrated_spectrum[emission_channel-C:emission_channel+C+1]
                continuum = np.concatenate((integrated_spectrum[:emission_channel-C-1], np.repeat(np.nan, int(2*C+1)), integrated_spectrum[emission_channel+C:]))
    
                fig, ax = plt.subplots(figsize=(12.8, 7.20))
                offset_vertical = np.nanmax(new_spectrum)*1.5
                plt.bar(full_channels, integrated_spectrum, color='#264653', label="Original spectrum", bottom=offset_vertical)
                plt.plot(full_channels, fitted_continuum+offset_vertical, color='#D62828', label="Fitted continuum", alpha=1)
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
                ax.set_ylabel(r"Flux density (%s)" %flux_units, labelpad=12.0)

                freq_axis = np.linspace(rest_freq-channel_to_freq*num_channels_cubelets/2, rest_freq+channel_to_freq*num_channels_cubelets, 2*num_channels_cubelets+1)*10**(-6)
                chann_axis = np.linspace(1, 2*num_channels_cubelets+1, 2*num_channels_cubelets+1)

                ax.grid(True, alpha=0.5, which="minor", ls=":")
                ax.grid(True, alpha=0.7, which="major")
                
                plt.bar(np.linspace(1, emission_channel-C-1, emission_channel-C-1), new_spectrum[:emission_channel-C-1])
                plt.bar(np.linspace(emission_channel-C, emission_channel+C, 2*C+1), new_central_region)
                plt.bar(np.linspace(emission_channel+C+1, 2*num_channels_cubelets+1, 2*num_channels_cubelets+1 - emission_channel - C), new_spectrum[emission_channel+C:])
                
                ax.vlines(emission_channel, min(spectrum_central_region), max(spectrum_central_region), linestyles='dashdot', color='red', label='HI position', alpha=1, zorder=0)
                
                #plt.plot(full_channels, gauss_function(full_channels, amplitud, media, sigma))

                ax.legend(loc='best')

                plt.tight_layout()

                plt.savefig(f'Verification_process/SN_best/regions_SN_spectrum_{L}_{C}.pdf')
            index_array_C += 1
        index_array_L += 1

    fig, ax = plt.subplots(figsize=(9, 7.20))
    
    centers = [1, num_channels_cubelets-1, 1, num_pixels_cubelets]
    dx, = np.diff(centers[:2])/(signal_to_noise_ratios.shape[1]-0.5)
    dy, = -np.diff(centers[2:])/(signal_to_noise_ratios.shape[0]-0.5)
    extent = [0.5, num_channels_cubelets+0.5, 0.5, num_pixels_cubelets+0.5]

    signals = ax.imshow(signal_to_noise_ratios, cmap="plasma", extent=extent, origin='lower', interpolation='nearest', aspect='auto')#, norm=LogNorm())

    plt.colorbar(signals)

    #?Figure's labels
    ax.set_xlabel("Value of C")
    ax.set_ylabel("Value of L")
    ax.set_title("S/N estimation in function of (C, L)", fontsize=20)

    plt.tight_layout()

    plt.savefig("Verification_process/SN_best/signal_to_noise.pdf")
    return L_best, C_best, S_N_best
