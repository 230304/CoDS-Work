import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

def load_shapefile(shapefile_path):
    """
    Load a shapefile using GeoPandas.

    Parameters:
        shapefile_path (str): Path to the shapefile.

    Returns:
        GeoDataFrame: Loaded shapefile as a GeoDataFrame.
    """
    return gpd.read_file(shapefile_path)

def mask_raster_with_shapefile(raster_file, shapefile_geometry):
    """
    Mask a raster file using the geometry from a shapefile.

    Parameters:
        raster_file (str): Path to the raster file.
        shapefile_geometry (geometry): Geometry from the shapefile to use for masking.

    Returns:
        tuple: Masked data and the associated affine transform.
    """
    with rasterio.open(raster_file) as src:
        # Mask the raster data using the shapefile geometry
        masked_data, masked_transform = mask(src, shapefile_geometry, crop=True)
        return masked_data, masked_transform

def plot_masked_data(masked_data, masked_transform):
    """
    Plot the masked raster data.

    Parameters:
        masked_data (ndarray): The masked data array.
        masked_transform (Affine): The transform associated with the masked data.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_data[0], extent=(masked_transform[2], 
                                        masked_transform[2] + masked_transform[0] * masked_data.shape[2], 
                                        masked_transform[5] + masked_transform[4] * masked_data.shape[1], 
                                        masked_transform[5]))
    plt.colorbar(label='Ground Water Storage Anomaly (GWSA)')
    plt.title('Masked GRACE GWSA Data over the State')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

if __name__ == "__main__":
    # Define file paths
    gwsa_file = r'/path/to/your/data/GWSA_TWSA_200204_cm_CSR_0.25_MASCON_LM.tif'
    shapefile_path = r'/path/to/your/data/Sgapefile.shp'
    
    # Load the shapefile of Uttar Pradesh
    up_shapefile = load_shapefile(shapefile_path)
    
    # Mask the GWSA raster data using the shapefile
    masked_data, masked_transform = mask_raster_with_shapefile(gwsa_file, up_shapefile.geometry)
    
    # Plot the masked GWSA data
    plot_masked_data(masked_data, masked_transform)
