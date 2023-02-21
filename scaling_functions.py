import numpy as np
from scipy import ndimage
from alive_progress import alive_bar
import matplotlib.pyplot as plt


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
    with alive_bar(num_galaxies, bar='circles', title='Spatial scaling of cubelets in progress') as bar:                    
        for i, cubelet in enumerate(cubelets):
            scale = (2*num_pixels_cubelets_wanted+1)/(2*num_pixels_cubelets[i]+1)
            scaled_cubelet = ndimage.geometric_transform(cubelet, scale_2D_image, cval=0, extra_keywords={'scale':scale, 'shift_x':0, 'shift_y':0}) #Rescale

            scaled_crop_cubelet = scaled_cubelet[:, :2*num_pixels_cubelets_wanted+1, :2*num_pixels_cubelets_wanted+1] #Crop

            scaled_cubelets.append(scaled_crop_cubelet)
            
            """print(f"Integrated flux ratio (new/old): {np.nansum(cubelet[0])/(scale**2*np.nansum(cubelet[0]))}")
            print(f"Maximum value ratio (new/old): {np.nanmax(scaled_crop_cubelet[0])/np.nanmax(cubelet[0])}")"""

            #!!! Keep track of the integrated flux changes

            """if(i==2):
                print(f"Integrated flux ratio: {np.nansum(scaled_crop_cubelet[17])/(scale**2*np.nansum(cubelet[17]))}.")
                print(f"Maximum value: {np.nanmax(spaxel)/np.nanmax(scaled_crop_spaxel)}.")
                f, axarr = plt.subplots(1,3)
                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                axarr[0].imshow(cubelet[17], origin='lower')
                axarr[1].imshow(scaled_cubelet[17], origin='lower')
                axarr[2].imshow(scaled_crop_cubelet[17], origin='lower')
                plt.show()"""
            bar()

    return scaled_cubelets


    #print(cubelets[0][0].shape)

    #new_image = (ndimage.geometric_transform(cubelets[0][0], scale_2D_image, cval=0, extra_keywords={'scale':scale, 'shift_x':0, 'shift_y':0})) #Rescale
    diff = -1
    scaled_cubelets = []
    for i in range(num_galaxies):

        scale = spatial_scales[i]

        print('\n', scale, 19/cubelets[i][0].shape[1], 19/21)

        print(cubelets[i][0].shape)

        new_image = resize(cubelets[i][0], (19, 19), anti_aliasing=False, preserve_range=False)/(19/cubelets[i][0].shape[1])**2
        #new_image = ndimage.geometric_transform(cubelets[i][0], scale_2D_image, cval=0, extra_keywords={'scale':19/cubelets[i][0].shape[1], 'shift_x':2, 'shift_y':2})[1:19, 1:19]/(19/cubelets[i][0].shape[1])**2

        #print(new_image)

        cropped_new_image = new_image[:2*num_pixels_cubelets_wanted+1, :2*num_pixels_cubelets_wanted+1]

        """print(f"Integrated flux ratio (new/old): {np.nanmean(new_image)/(np.nanmean(cubelets[i][0]))}")

        if(abs(np.nansum(cropped_new_image)/(scale**2*np.nansum(np.abs(cubelets[i][0]))) - 1) > diff):
            diff = abs(np.nansum(cropped_new_image)/(scale**2*np.nansum(cubelets[i][0])) - 1)
        
        print(f"Maximum value ratio (new/old): {np.nanmax(cropped_new_image)/np.nanmax(cubelets[i][0])}")"""

        #!!! Keep track of the integrated flux changes
        """if(abs(np.nansum(cropped_new_image)/(scale**2*np.nansum(cubelets[i][0])) - 1) > 0.89):
            factor = np.nansum(np.nansum(new_image)/(scale**2*np.nansum(cubelets[0][0])))
            f, axarr = plt.subplots(1, 2)
            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            axarr[0].imshow(cubelets[i][0], origin='lower')
            axarr[1].imshow(new_image, origin='lower')
            plt.show()
        print(diff)"""
        scaled_cubelets.append(cropped_new_image)

    return scaled_cubelets

def spectral_scaling(num_galaxies, num_channels_cubelets, num_channels_cubelets_wanted, cubelets, spectral_scales):
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

    scaled_cubelets = np.zeros((num_galaxies, 2*num_channels_cubelets_wanted+1, cubelets[0].shape[1], cubelets[0].shape[2]))
    with alive_bar(num_galaxies*cubelets[0].shape[2]*cubelets[0].shape[1], bar='circles', title='Spectral scaling of cubelets in progress') as bar:
        for i, cubelet in enumerate(cubelets):
            scale = (2*num_channels_cubelets_wanted+1)/(2*num_channels_cubelets[i]+1)
            for x in range(cubelet.shape[2]):
                for y in range(cubelet.shape[1]):
                    #spaxel = 100*stats.norm.pdf(np.linspace(1, 100, 100), 50, np.sqrt(300))
                    spaxel = cubelet[:, y, x]

                    spaxel_matrix = np.stack((np.zeros(len(spaxel)), spaxel)) #We create a matrix in order to use 'ndimage.geometric_transform'

                    scaled_spaxel_matrix = ndimage.geometric_transform(spaxel_matrix, scale_1D_spectrum, cval=0, extra_keywords={'scale':scale, 'shift':0}) #Rescale

                    scaled_spaxel = scaled_spaxel_matrix[1] #We extract the spectrum only
                    
                    scaled_crop_spaxel = scaled_spaxel[:2*num_channels_cubelets_wanted+1] #Crop

                    """print(f"Integrated flux ratio (new/old): {np.nansum(scaled_crop_spaxel)/(scale*np.nansum(spaxel))}")
                    print(f"Maximum value ratio (new/old): {np.nanmax(scaled_crop_spaxel)/np.nanmax(spaxel)}")"""

                    """if(i==2 and x==10, y==10):
                        f, axarr = plt.subplots(3,1) 
                        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                        axarr[0].plot(np.linspace(1, len(spaxel), len(spaxel)), spaxel)
                        axarr[1].plot(np.linspace(1, len(scaled_spaxel), len(scaled_spaxel)), scaled_spaxel)
                        axarr[2].plot(np.linspace(1, len(scaled_crop_spaxel), len(scaled_crop_spaxel)), scaled_crop_spaxel)
                        plt.show()
                        exit()"""
                    bar()

                    scaled_cubelets[i, :, y, x] = scaled_crop_spaxel

    return scaled_cubelets

"""from astropy.utils.data import download_file

import matplotlib.pyplot as plt

from astropy.io import fits
image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )

hdu_list = fits.open(image_file)
hdu_list.info()
datos = hdu_list[0].data[210:231, 210:231]

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

image = color.rgb2gray(data.astronaut())

print(image.shape)
print(datos.shape)

new_data = resize(datos, (19, 19), anti_aliasing=True, preserve_range=True)

print(type(datos))

print(new_data)

scale = 19/21

print(f"Integrated flux ratio (new/old): {np.nansum(new_data)/(scale**2*np.nansum(datos))}")

print(f"Maximum value ratio (new/old): {np.nanmax(new_data)/np.nanmax(datos)}")

f, axarr = plt.subplots(1, 2)
# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(datos, origin='lower')
axarr[1].imshow(new_data, origin='lower')
plt.show()

"""