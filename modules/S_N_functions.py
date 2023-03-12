from modules.functions import extract_spectrum_from_spatial_circular_region, fit_continuum_of_spectrum
import numpy as np
import matplotlib.pyplot as plt


def S_N_calculation(datacube: np.ndarray, num_channels: int, center_x: int, center_y: int,
                    emission_channel: int, L: int, C: int, degree_fit_continuum: int) -> float:
    """
    Calculates the signal-to-noise ratio of a given spectrum.

    Parameters
    ----------
    datacube : numpy.ndarray
        The 3D datacube containing the spectral data.
    num_channels : int
        The number of spectral channels to extract.
    center_x : int
        The x coordinate of the center of the circular region to extract the spectrum from.
    center_y : int
        The y coordinate of the center of the circular region to extract the spectrum from.
    emission_channel : int
        The index of the emission channel to be used as reference.
    L : int
        The radius of the circular region to extract the spectrum from, in pixels.
    C : int
        The number of channels to include in the central region around the emission channel.
    degree_fit_continuum : int
        The degree of the polynomial to use when fitting the continuum.

    Returns
    -------
    float
        The signal-to-noise ratio of the spectrum.
    """

    spectrum = extract_spectrum_from_spatial_circular_region(datacube, center_x, center_y, L)
    
    # Fit continuum of spectrum
    x_axis = np.arange(2 * num_channels + 1)
    new_spectrum, fitted_central_region, fitted_continuum = fit_continuum_of_spectrum(spectrum, x_axis, emission_channel, C, degree_fit_continuum)
    std_dev = np.nanstd(fitted_continuum)
    S_N = np.nansum(fitted_central_region) / (std_dev * np.sqrt(2 * C + 1))

    return S_N

def S_N_measurement_test(datacube, num_pixels_cubelets, num_channels_cubelets, center_x, center_y, emission_channel, degree_fit_continuum):
    """
    Signal-to-noise measurement test.

    Parameters
    ----------
    datacube : numpy.ndarray
        The data cube.
    num_pixels_cubelets : int
        The number of pixels of the cubelets.
    num_channels_cubelets : int
        The number of channels of the cubelets.
    center_x : int
        The central pixel of the x-axis.
    center_y : int
        The central pixel of the y-axis.
    emission_channel : int
        The channel number of the emission line.
    rest_freq : float
        The rest frequency of the emission line.
    channel_to_freq : float
        The conversion factor from channel to frequency.
    flux_units : str
        The units of the flux density.
    degree_fit_continuum : int
        The degree of the polynomial function used to fit the continuum.

    Returns
    -------
    signal_to_noise_ratios : numpy.ndarray
        The signal-to-noise ratios for each cubelet.
    """
    S_N_best = -1
    L_best, C_best = 0, 0
    L_max, C_max = num_pixels_cubelets+1, int(datacube.shape[0]/4)+1
    signal_to_noise_ratios = np.zeros((L_max-1, C_max-1))
    index_array_L = 0
    for L in range(1, num_pixels_cubelets+1):
        integrated_spectrum = extract_spectrum_from_spatial_circular_region(datacube, center_x, center_y, L)
        index_array_C = 0
        for C in range(1, C_max):
            x_axis = np.arange(2*num_channels_cubelets+1)
            _, fitted_central_region, fitted_continuum = fit_continuum_of_spectrum(integrated_spectrum, x_axis, emission_channel, C, degree_fit_continuum)

            std_dev = np.nanstd(fitted_continuum)

            S_N = np.nansum(fitted_central_region)/(std_dev*np.sqrt(2*C+1))

            if S_N >= S_N_best:
                L_best, C_best = L, C
                S_N_best = S_N
            signal_to_noise_ratios[index_array_L, index_array_C] = S_N

            index_array_C += 1
        index_array_L += 1

    fig, ax = plt.subplots(figsize=(9, 7.20))

    centers = [1, C_max - 1, 1, L_max]
    dx, = np.diff(centers[:2]) / (signal_to_noise_ratios.shape[1] - 0.5)
    dy, = -np.diff(centers[2:]) / (signal_to_noise_ratios.shape[0] - 0.5)
    extent = [0.5, C_max + 0.5, 0.5, L_max + 0.5]

    signals = ax.matshow(signal_to_noise_ratios, cmap="plasma", extent=extent, origin='lower', interpolation='nearest', aspect='auto')

    plt.colorbar(signals)

    #?Figure's labels
    ax.set_xlabel("Value of C")
    ax.set_ylabel("Value of L")
    ax.set_title("S/N estimation in function of (C, L)", fontsize=20)

    plt.tight_layout()

    fig.savefig("Verification_process/SN_best/Signal_to_noise_ratios.pdf")

    return S_N_best, L_best, C_best