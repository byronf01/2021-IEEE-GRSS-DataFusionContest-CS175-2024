""" This module contains functions for plotting satellite images. """
import re
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import (
    minmax_scale,
    quantile_clip
)
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here
    
    stack = viirs_stack.flatten()
    plt.hist(stack, bins=n_bins, log=True)
    plt.xlabel("viirs vals")
    plt.ylabel("frequency")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()



def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here

    tiles, times, bands, x, y = sentinel1_stack.shape
    reshaped_data = sentinel1_stack.reshape(tiles, times, bands, -1)
    histogram_values_1 = reshaped_data[..., 0, :]  
    histogram_values_2 = reshaped_data[..., 1, :]   
    histogram_values_1 = histogram_values_1.flatten()
    histogram_values_2 = histogram_values_2.flatten()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].hist(histogram_values_1, bins=n_bins, log=True)
    ax[0].set_title('sentinel VH')
    ax[1].hist(histogram_values_2, bins=n_bins, log=True)
    ax[1].set_title('sentinel VV')

  
    ax[0].set_xlabel('sentinel values')
    ax[0].set_ylabel('frequency')
    ax[1].set_xlabel('sentinel values')
    ax[1].set_ylabel('frequency')

    
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()
  


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    
    """
    stack = preprocess_data(sentinel2_stack, "sentinel1")
    c = stack.flatten()
    plt.hist(c, log=True, bins=n_bins)
    """

    tiles, times, bands, x, y = sentinel2_stack.shape
    reshaped_data = sentinel2_stack.reshape(tiles, times, bands, -1)
    fig, ax = plt.subplots(4, 3, tight_layout=True, figsize=(27, 20))

    for band, name in zip(range(bands), metadata[0][0].bands):
        data = reshaped_data[..., band, :].flatten()
        ax[band // 3][band % 3].hist(data, bins=n_bins, log=True)
        ax[band // 3][band % 3].set_title(f'{metadata[0][0].satellite_type} {name}')
        ax[band // 3][band % 3].set_xlabel(f'{metadata[0][0].satellite_type} values')
        ax[band // 3][band % 3].set_ylabel('frequency')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
  
    tiles, times, bands, x, y = landsat_stack.shape
    reshaped_data = landsat_stack.reshape(tiles, times, bands, -1)
    fig, ax = plt.subplots(4, 3, tight_layout=True, figsize=(27, 20))

    for band, name in zip(range(bands), metadata[0][0].bands):
        data = reshaped_data[..., band, :].flatten()
        ax[band // 3][band % 3].hist(data, bins=n_bins, log=True)
        ax[band // 3][band % 3].set_title(f'{metadata[0][0].satellite_type} {name}')
        ax[band // 3][band % 3].set_xlabel(f'{metadata[0][0].satellite_type} values')
        ax[band // 3][band % 3].set_ylabel('frequency')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()


def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    c = ground_truth.flatten()
    plt.hist(c)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()


def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    plt.imshow(viirs, cmap='viridis')
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()
    


def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    viirs = preprocess_data(viirs_stack, "viirs")
    fig, ax = plt.subplots(1, len(viirs),tight_layout=True, figsize=(20, 8))
    for i, date in enumerate(viirs):
        for band in date:
            hist = ax[i].imshow(band, cmap='viridis')
            
            ax[i].set_title(f'{metadata[i].satellite_type} @ {metadata[i].time}')
    plt.axis('off')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()
    


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    f = {"sentinel1": preprocess_sentinel1, "sentinel2": preprocess_sentinel2, "landsat": preprocess_landsat,
        "viirs": preprocess_viirs}
    return f[satellite_type](satellite_stack)


def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """

    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """

    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")
   
    # fill in the code here
    fig, ax = plt.subplots(1, len(processed_stack),tight_layout=True, figsize=(22, 8))
    for i, time in enumerate(processed_stack):
        vh, vv, vb = time[0], time[1], time[1] - time[0]
        vh, vv, vb = vh[np.newaxis, np.newaxis, ...], vv[np.newaxis, np.newaxis, ...], vb[np.newaxis, np.newaxis, ...]
        r, g, b = minmax_scale(quantile_clip(vv, 0.02)), minmax_scale(quantile_clip(vh, 0.02)), minmax_scale(quantile_clip(vb, 0.02))
        rgb = np.dstack((r[0][0], g[0][0], b[0][0]))

        ax[i].imshow(rgb)
        ax[i].set_title(f'{metadata[i].satellite_type} @ {metadata[i].time}, VV VH VV/VH')


    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()



def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    for group in bands_to_plot:
        for band in group:
            # set to correct indice how??
            if band not in band_mapping: raise KeyError("Invalid band")
    return None


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
   
    # fill in the code here
    fig, ax = plt.subplots(len(bands_to_plot), len(processed_stack), figsize=(20, 15)) # bands to plot by all times
    for i, comb in enumerate(bands_to_plot):
        for j, time in enumerate(processed_stack):
            # are filters and corrections needed?
            n1, n2, n3 = band_mapping[comb[0]], band_mapping[comb[1]], band_mapping[comb[2]]
            a, b, c = time[n1], time[n2], time[n3]
            ax[i][j].imshow(np.dstack((a, b, c)))
            ax[i][j].set_title(f'{metadata[i].satellite_type} @ {metadata[i].time}, {" ".join(comb)}')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(
            # Path(image_dir) / f"plot_{metadata[0][0]}.png"
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()



def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """

    # Using a list of lists to dynamically group bands by date
    """
    d = {}
    for f in files:
        f = f[4:]
        date = re.findall(r'^(.+?)_', f)[0]
        band = re.findall(r'_(.+)', f)[0]
        if date not in d: 
            d[date] = []
        d[date].append(band)
    """
    return [m.bands for m in metadata]




def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    # fill in the code here
    plt.imshow(ground_truth[0][0], cmap='viridis')
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()


    