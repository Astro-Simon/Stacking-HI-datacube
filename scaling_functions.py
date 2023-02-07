import numpy as np
from scipy import ndimage
from alive_progress import alive_bar
import matplotlib.pyplot as plt

from spectres import spectres
import scipy.stats as stats
import math


def scale_2D_image(output_coords, scale, shift_x, shift_y):
    dim_y = output_coords[1]
    dim_x = output_coords[2]

    rescaled_dim_x = dim_x/scale
    rescaled_dim_y = dim_y/scale

    shifted_rescaled_dim_x = rescaled_dim_x + shift_x*(1-1/scale)
    shifted_rescaled_dim_y = rescaled_dim_y + shift_y*(1-1/scale)

    return (output_coords[0], shifted_rescaled_dim_y, shifted_rescaled_dim_x)

def scale_1D_spectrum(output_coords, scale, shift):
    dim_z = output_coords[1]

    rescaled_dim_z = dim_z/scale

    shifted_rescaled_dim_z = rescaled_dim_z + shift*(1-1/scale)

    return (output_coords[0], shifted_rescaled_dim_z)

#!!! One scaling for each cubelets or one scaling for each channel??
def spatial_scaling(num_galaxies, num_pixels_cubelets, cubelets):
    """
    Function that calculate the spatial scaling necessary for each cubelet and scale them.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - redshifts [array - float]: Redshifts of all the galaxies
    - cubelets [array - float]: Array of cubelets of each galaxy

    • Output
    - scaled_cubelets [array - float]: Array of cubelets of each galaxy scaled spatially
    """
    scaled_cubelets = []
    with alive_bar(num_galaxies, bar='circles', title='Spatial scaling of cubelets in progress') as bar:                    
        for i, cubelet in enumerate(cubelets):
            scale = np.nanmin(num_pixels_cubelets)/num_pixels_cubelets[i]
            scaled_cubelet = ndimage.geometric_transform(cubelet, scale_2D_image, cval=0, extra_keywords={'scale':scale, 'shift_x':0, 'shift_y':0})

            scaled_crop_cubelet = scaled_cubelet[:, :2*np.nanmin(num_pixels_cubelets)+1, :2*np.nanmin(num_pixels_cubelets)+1]

            scaled_cubelets.append(scaled_crop_cubelet)

            #!!! Keep track of the integrated flux changes 

            """if(i==2):
                f, axarr = plt.subplots(1,3) 
                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                axarr[0].imshow(cubelet[17], origin='lower')
                axarr[1].imshow(scaled_cubelet[17], origin='lower')
                axarr[2].imshow(scaled_crop_cubelet[17], origin='lower')
                plt.show()"""
            bar()

    return scaled_cubelets

def spectral_scaling(num_galaxies, num_channels_cubelets, cubelets):
    """
    Function that calculate the spectral scaling necessary for each cubelet and scale them.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Number of channels in spectral axis
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - redshifts [array - float]: Redshifts of all the galaxies
    - cubelets [array - float]: Array of cubelets of each galaxy

    • Output
    - scaled_cubelets [array - float]: Array of cubelets of each galaxy scaled spatially
    """

    """mu = 50
    variance = 300
    sigma = math.sqrt(variance)
    old = 100
    new = 150
    range_old = np.linspace(0, old, old)
    range_new = np.linspace(0, new, new)
    gauss_old = 100*stats.norm.pdf(range_old, mu, sigma)
    from scipy.stats import binned_statistic

    
    #range_new = np.linspace(0, len(cubelets[239][:, 10, 11]), len(cubelets[239][:, 10, 11]))
    #range_old = np.linspace(0, len(cubelets[91][:, 10, 11]), len(cubelets[91][:, 10, 11]))
    
    scaled_spectrum = spectres(range_new, range_old, gauss_old, fill=3)
    print(np.nansum(gauss_old))
    print(np.nansum(scaled_spectrum))

    print(f"old:{len(cubelets[91][:, 10, 11])}, new:{len(cubelets[239][:, 10, 11])}")

    f, axarr = plt.subplots(2,1)
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].plot(range_old, gauss_old)
    axarr[1].plot(range_new, scaled_spectrum)

    plt.show()"""

    scaled_cubelets = np.zeros((num_galaxies, 2*np.nanmin(num_channels_cubelets)+1, cubelets[0].shape[1], cubelets[0].shape[2]))    
    with alive_bar(num_galaxies*cubelets[0].shape[2]*cubelets[0].shape[1], bar='circles', title='Spectral scaling of cubelets in progress') as bar:                    
        for i, cubelet in enumerate(cubelets):
            scale = np.nanmin(num_channels_cubelets)/num_channels_cubelets[i]
            for x in range(cubelet.shape[2]):
                for y in range(cubelet.shape[1]):
                    #spaxel = 100*stats.norm.pdf(np.linspace(1, 100, 100), 50, np.sqrt(300))
                    spaxel = cubelet[:, y, x]

                    spaxel_matrix = np.stack((np.zeros(len(spaxel)), spaxel))

                    scaled_spaxel_matrix = ndimage.geometric_transform(spaxel_matrix, scale_1D_spectrum, cval=0, extra_keywords={'scale':scale, 'shift':0})

                    scaled_spaxel = scaled_spaxel_matrix[1]
                    
                    scaled_crop_spaxel = scaled_spaxel[:2*np.nanmin(num_channels_cubelets)+1]

                    scaled_cubelets[i, :, y, x] = scaled_crop_spaxel

                    """if(i==1 and x==10, y==10):
                        print(scaled_spaxel_matrix)
                        f, axarr = plt.subplots(3,1) 
                        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                        axarr[0].plot(np.linspace(1, len(spaxel), len(spaxel)), spaxel)
                        axarr[1].plot(np.linspace(1, len(scaled_spaxel), len(scaled_spaxel)), scaled_spaxel)
                        axarr[2].plot(np.linspace(1, len(scaled_crop_spaxel), len(scaled_crop_spaxel)), scaled_crop_spaxel)
                        plt.show()
                        exit()"""
                    bar()

    return scaled_cubelets


        