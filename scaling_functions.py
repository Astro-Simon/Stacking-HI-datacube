import numpy as np
from scipy import ndimage
from alive_progress import alive_bar


def scale_2D_image(output_coords, scale, shift_x=0, shift_y=0):
    """
    Scale the 2D image using the given scale and shifts.

    Parameters:
    -----------
    output_coords : tuple
        Tuple of the output coordinates (dim_y, dim_x).

    scale : float
        Scale factor.

    shift_x : float
        Shift factor in the x direction.

    shift_y : float
        Shift factor in the y direction.

    Returns:
    --------
    tuple
        Tuple of the rescaled and shifted output coordinates (shifted_rescaled_dim_y, shifted_rescaled_dim_x).
    """
    dim_y, dim_x = output_coords

    rescaled_dim_x = dim_x / scale
    rescaled_dim_y = dim_y / scale

    shifted_rescaled_dim_x = rescaled_dim_x + shift_x * (1 - 1 / scale)
    shifted_rescaled_dim_y = rescaled_dim_y + shift_y * (1 - 1 / scale)

    return (shifted_rescaled_dim_y, shifted_rescaled_dim_x)


def scale_1D_spectrum(output_coords, scale, shift=0):
    """
    Scale the 1D spectrum using the given scale and shift.

    Parameters:
    -----------
    output_coords : tuple
        Tuple of the output coordinates (dim_z,).

    scale : float
        Scale factor.

    shift : float
        Shift factor.

    Returns:
    --------
    tuple
        Tuple of the rescaled and shifted output coordinates (shifted_rescaled_dim_z,).
    """
    dim_z, = output_coords

    rescaled_dim_z = dim_z / scale

    shifted_rescaled_dim_z = rescaled_dim_z + shift * (1 - 1 / scale)

    return (shifted_rescaled_dim_z,)



def spatial_scaling(num_galaxies, num_pixels_cubelets, num_pixels_cubelets_wanted, cubelets):
    """
    Rescale the spatial dimensions of cubelets.

    Parameters:
    -----------
    num_galaxies : int
        Number of galaxies in the cubelets set.
    num_pixels_cubelets : array
        Array of integers representing the number of pixels of each cubelet.
    num_pixels_cubelets_wanted : int
        Desired number of pixels of the cubelets.
    cubelets : array
        Set of cubelets to be rescaled.

    Returns:
    --------
    scaled_cubelets : array
        Rescaled and cropped cubelets.

    """
    scaled_cubelets = []
    with alive_bar(num_galaxies, bar='circles', title='Spatial scaling of cubelets in progress') as bar:
        for i, cubelet in enumerate(cubelets):
            if(num_pixels_cubelets_wanted < num_pixels_cubelets[i]):
                scale = int(2 * num_pixels_cubelets_wanted + 1) / int(2 * num_pixels_cubelets[i] + 1)
                scaled_cropped_cubelet = np.zeros((cubelet.shape[0], 2 * num_pixels_cubelets_wanted + 1, 2 * num_pixels_cubelets_wanted + 1))
                for z in range(cubelet.shape[0]):
                    scaled_cubelet = ndimage.geometric_transform(cubelet[z], scale_2D_image, cval=0, extra_keywords={'scale': scale, 'shift_x': 0, 'shift_y': 0})  # Rescale
                    scaled_cropped_cubelet[z] = scaled_cubelet[:2 * num_pixels_cubelets_wanted + 1, :2 * num_pixels_cubelets_wanted + 1]  # Crop
                scaled_cubelets.append(scaled_cropped_cubelet)
            elif(num_pixels_cubelets_wanted > num_pixels_cubelets[i]):
                print("\nERROR: we found a cubelet with less pixels than the wanted number.\n")
            else: #(num_pixels_cubelets_wanted == num_pixels_cubelets[i])
                scaled_cubelets.append(cubelet)
            bar()

    return scaled_cubelets


def spectral_scaling(num_galaxies, num_channels_cubelets, num_channels_cubelets_wanted, num_pixels_cubelets_wanted, cubelets):
    """
    Rescale the spectral dimension of cubelets.

    Parameters:
    -----------
    num_galaxies : int
        Number of galaxies.

    num_channels_cubelets : list
        List of number of channels of the cubelets.

    num_channels_cubelets_wanted : int
        Number of channels wanted.

    num_pixels_cubelets_wanted : int
        Number of pixels wanted.

    cubelets : numpy array
        Array of cubelets.

    Returns:
    --------
    numpy array
        Array of rescaled cubelets.
    """
    
    scaled_cubelets = np.zeros((num_galaxies, 2 * num_channels_cubelets_wanted + 1, 2 * num_pixels_cubelets_wanted + 1, 2 * num_pixels_cubelets_wanted + 1))

    with alive_bar(num_galaxies, bar='circles', title='Spectral scaling of cubelets in progress') as bar:
        for i, cubelet in enumerate(cubelets):
            if(num_channels_cubelets_wanted < num_channels_cubelets[i]):
                scale = int(2*num_channels_cubelets_wanted+1)/int(2*num_channels_cubelets[i]+1)
                for x in range(cubelet.shape[2]):
                    for y in range(cubelet.shape[1]):
                        spectrum = cubelet[:, y, x]
                        scaled_spectrum = ndimage.geometric_transform(spectrum, scale_1D_spectrum, cval=0, extra_keywords={'scale':scale, 'shift':0}) # Rescale
                        scaled_crop_spectrum = scaled_spectrum[:2*num_channels_cubelets_wanted+1] # Crop
                        scaled_cubelets[i, :, y, x] = scaled_crop_spectrum
            elif(num_channels_cubelets_wanted > num_channels_cubelets[i]):
                print("\nERROR: we found a cubelet with less channels than the wanted number.\n")
            else: #(num_channels_cubelets_wanted == num_channels_cubelets[i])
                scaled_cubelets[i] = cubelet
            bar()

                    
    return scaled_cubelets

