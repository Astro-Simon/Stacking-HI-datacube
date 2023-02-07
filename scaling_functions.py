import numpy as np
from scipy import ndimage
from alive_progress import alive_bar
import matplotlib.pyplot as plt
def scale_2D_image(output_coords, scale, shift_x, shift_y):
    dim_y = output_coords[1]
    dim_x = output_coords[2]

    rescaled_dim_x = dim_x*scale
    rescaled_dim_y = dim_y*scale

    shifted_rescaled_dim_x = rescaled_dim_x + shift_x*(1-scale)
    shifted_rescaled_dim_y = rescaled_dim_y + shift_y*(1-scale)

    return (output_coords[0], shifted_rescaled_dim_y, shifted_rescaled_dim_x)


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
            scale = np.nanmax(num_pixels_cubelets)/num_pixels_cubelets[i]
            scaled_cubelet = ndimage.geometric_transform(cubelet, scale_2D_image, cval=0, extra_keywords={'scale':scale, 'shift_x':0, 'shift_y':0})

            scaled_crop_cubelet = scaled_cubelet[:, :2*np.nanmin(num_pixels_cubelets)+1, :2*np.nanmin(num_pixels_cubelets)+1]

            scaled_cubelets.append(scaled_crop_cubelet)
            



            if(i==1):
                f, axarr = plt.subplots(1,3) 
                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                axarr[0].imshow(cubelet[10], origin='lower')
                axarr[1].imshow(scaled_cubelet[10], origin='lower')
                axarr[2].imshow(scaled_crop_cubelet[10], origin='lower')
                plt.show()
            bar()
            
    exit()
    return scaled_cubelets

def test():
    # create 4 * 4 dim array.
    b = np.arange(16).reshape((4, 4))
    
    # reducing dimensions function
    def shift_func(output_coords):
        return (output_coords[0], output_coords[1]-1)
    
    
    b_finish = ndimage.geometric_transform(b, shift_func)

    print(b)
    print(b_finish)

def spectral_scaling(num_galaxies, pixel_scale, num_pixels_cubelets, rest_freq_HI, freq_ini, channel_to_freq, redshifts, cubelets):
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

    greatest_scale_length = -1
    channel_furthest_galaxy = -1
    scale_lengths = np.zeros(num_galaxies)
    # First we calculate the factor needed for each spec 
    for index, redshift in enumerate(redshifts):
        HI_positions = rest_freq_HI/(1+redshift) #* Frequency of the position of the HI line (spectral observational position - not at rest)
        channels_HI = int((HI_positions - freq_ini)/channel_to_freq) #* Channel of the HI line (spectral observational position - not at rest)
        #!!! In order to calculate the factor I will use the non-scaled-spectrally positions of the galaxies...

        print(type(redshift))

        if(channels_HI > channel_furthest_galaxy):
            channel_furthest_galaxy = channels_HI #* We calculate the channel where the furthest galaxy lies
        pixel_scale = pixel_scale*u.deg
        scale_lengths[index] = (cosmo.luminosity_distance(redshift) * pixel_scale).value

        if(scale_lengths[index] > greatest_scale_length):
            greatest_scale_length = scale_lengths[index]

    scaling_factors = greatest_scale_length/scale_lengths
    scaled_nums_pixels_cubelets = scaling_factors*num_pixels_cubelets
    print(scaled_nums_pixels_cubelets)

    for i, cubelet in enumerate(cubelets):
        scaled_cubelets = ndimage.geometric_transform(cubelet, scale_image, cval=0, extra_keywords={'scale':scaling_factors[i]}) #!!! How do I use it?
