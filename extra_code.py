import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = data[0][1].shape

    image_HR = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    for index, value in enumerate(data):
        image_HR[index,:,:,:], image_LR[index,:,:,:], _ = value

    return image_HR, image_LR

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = torch.unsqueeze(data[0][1][0],0).shape

    image_HR = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    latlon_HR = np.empty((data_size, 4, 2))
    latlon_LR = np.empty((data_size, 4, 2))
    time_HR = torch.empty(data_size)
    time_LR = torch.empty(data_size)
    for index, value in enumerate(data):
        image_HR[index,:,:,:] = value[0][0]
        image_LR[index,:,:,:] = torch.unsqueeze(value[1][0],0)
        latlon_HR[index,:,:] = value[0][1]
        latlon_LR[index,:,:] = value[1][1]
        time_HR[index] = value[0][2]
        time_LR[index] = value[1][2]

    return image_HR, image_LR, latlon_HR, latlon_LR, time_HR, time_LR

# Plot large area map
polygons = []
pos = pos[:3]
for boundary in pos:
    polygon = Polygon(np.vstack((boundary[:,[1,0]][[0,2,3,1],:], boundary[0,[1,0]]))) # swap lat, lon + repeat first row
    polygons.append(polygon)

gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
gdf['sst'] = np.mean(np.array(sst_image), axis=(1,2)) # georeferencing patch is difficult

# Plot the GeoDataFrame
world_gdf = gpd.read_file('110m_cultural/ne_110m_admin_0_countries.shp') # Load the downloaded shapefile
fig, ax = plt.subplots(figsize=(10, 10))
world_gdf.plot(ax=ax, color='white', edgecolor='black')
gdf.plot(ax=ax, column='sst') #, cmap='viridis', legend=True)
plt.show()
plt.savefig(out_path + data_name + '_map.png', dpi=300, bbox_inches='tight')
