import numpy as np
from scipy import ndimage
from alive_progress import alive_bar
import matplotlib.pyplot as plt


def scale_2D_image(output_coords, scale, shift_x, shift_y):
    dim_y = output_coords[0]
    dim_x = output_coords[1]

    rescaled_dim_x = dim_x/scale
    rescaled_dim_y = dim_y/scale

    shifted_rescaled_dim_x = rescaled_dim_x + shift_x*(1-1/scale)
    shifted_rescaled_dim_y = rescaled_dim_y + shift_y*(1-1/scale)

    return (shifted_rescaled_dim_y, shifted_rescaled_dim_x)

def scale_1D_spectrum(output_coords, scale, shift):
    dim_z = output_coords[0]

    rescaled_dim_z = dim_z/scale

    shifted_rescaled_dim_z = rescaled_dim_z + shift*(1-1/scale)

    return (shifted_rescaled_dim_z,)

def spatial_scaling(num_galaxies, num_pixels_cubelets, num_pixels_cubelets_wanted, cubelets, spatial_scales):
    """
    Function that calculate the spatial scaling necessary for each cubelet and scale them.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Semirange of channels in spectral axis
    - num_channels_cubelets [int]: Semirange of channels in spectral axis
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - redshifts [array - float]: Redshifts of all the galaxies
    - cubelets [array - float]: Array of cubelets of each galaxy

    • Output
    - scaled_cubelets [array - float]: Array of cubelets of each galaxy scaled spatially
    """

    scaled_cubelets = []
    max_value_flux = -1
    with alive_bar(num_galaxies, bar='circles', title='Spatial scaling of cubelets in progress') as bar:                    
        for i, cubelet in enumerate(cubelets):
            scale = int(2*num_pixels_cubelets_wanted+1)/int(2*num_pixels_cubelets[i]+1)
            scaled_cropped_cubelet = np.zeros((cubelet.shape[0], 2*num_pixels_cubelets_wanted+1, 2*num_pixels_cubelets_wanted+1))
            for z in range(cubelet.shape[0]):
                scaled_cubelet = ndimage.geometric_transform(cubelet[z], scale_2D_image, cval=0, extra_keywords={'scale':scale, 'shift_x':0, 'shift_y':0}) #Rescale

                scaled_cropped_cubelet[z] = scaled_cubelet[:2*num_pixels_cubelets_wanted+1, :2*num_pixels_cubelets_wanted+1] #Crop

                """#!!! Keep track of the integrated flux changes
                scal2orig = cubelet[z].shape[0]/scaled_cropped_cubelet[z].shape[0] * cubelet[z].shape[1]/scaled_cropped_cubelet[z].shape[1]
                total_sum_original = np.nansum(cubelet[z])
                total_sum_scaled = np.nansum(scaled_cropped_cubelet[z])*scal2orig
                if(abs((total_sum_original-total_sum_scaled)/total_sum_original) > max_value_flux):    
                    max_value_flux = (total_sum_original-total_sum_scaled)/total_sum_original"""

            scaled_cubelets.append(scaled_cropped_cubelet)

            bar()

    #print(f"\nMaximum deviation in integrated flux: {max_value_flux*100:.2f}%.\n")

    return scaled_cubelets

def spectral_scaling(num_galaxies, num_channels_cubelets, num_channels_cubelets_wanted, num_pixels_cubelets_wanted, cubelets, spectral_scales):
    """
    Function that scales the spectral dimension of each cubelet.
    Function that scales the spectral dimension of each cubelet.

    • Input
    - num_galaxies [int]: Number of galaxies of the sample
    - num_channels_cubelets [int]: Semirange of channels in spectral axis
    - num_channels_cubelets [int]: Semirange of channels in spectral axis
    - rest_freq_HI [float]: Frequency around which spectra are shifted and wrapped
    - freq_ini [float]: Initial frequency (Hz)
    - channel_to_freq [float]: Ratio between channel and frequency
    - redshifts [array - float]: Redshifts of all the galaxies
    - cubelets [array - float]: Array of cubelets of each galaxy

    • Output
    - scaled_cubelets [array - float]: Array of cubelets of each galaxy scaled spatially

    """
    max_value_flux = -1
    first = True
    scaled_cubelets = np.zeros((num_galaxies, 2*num_channels_cubelets_wanted+1, 2*num_pixels_cubelets_wanted+1, 2*num_pixels_cubelets_wanted+1))
    with alive_bar(num_galaxies*(2*num_pixels_cubelets_wanted+1)*(2*num_pixels_cubelets_wanted+1), bar='circles', title='Spectral scaling of cubelets in progress') as bar:
        for i, cubelet in enumerate(cubelets):
            scale = int(2*num_channels_cubelets_wanted+1)/int(2*num_channels_cubelets[i]+1)
            for x in range(cubelet.shape[2]):
                for y in range(cubelet.shape[1]):
                    from scipy import stats
                    #spectrum = np.random.normal(0,100,len(cubelet[:, y, x]))#stats.norm.pdf(np.linspace(1, 100, len(cubelet[:, y, x])), 50, np.sqrt(300))
                    spectrum = cubelet[:, y, x]
                    
                    scaled_spectrum = ndimage.geometric_transform(spectrum, scale_1D_spectrum, cval=0, extra_keywords={'scale':scale, 'shift':0}) #Rescale
                    
                    scaled_crop_spectrum = scaled_spectrum[:2*num_channels_cubelets_wanted+1] #Crop
                    
                    """#!!! Keep track of the integrated flux changes
                    total_sum_original += np.nansum(spectrum)

                    scal2orig = len(spectrum)/len(scaled_crop_spectrum)

                    total_sum_scaled = np.nansum(scaled_crop_spectrum)*scal2orig

                    if(abs((total_sum_original-total_sum_scaled)/total_sum_original) > max_value_flux):    
                        max_value_flux = (total_sum_original-total_sum_scaled)/total_sum_original

                    if(max_value_flux > 20 and first == True):
                        plt.plot(np.arange(len(spectrum)), spectrum)
                        plt.plot(np.arange(len(scaled_crop_spectrum)), scaled_crop_spectrum)
                        plt.show()
                        first = False"""

                    scaled_cubelets[i, :, y, x] = scaled_crop_spectrum

                    bar()

    #print(f"\nMaximum deviation in integrated flux: {max_value_flux*100:.2f}%.\n")
    
    return scaled_cubelets
