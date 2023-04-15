import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# load data
kiva_loans = pd.read_csv("kiva_loans.csv", error_bad_lines=False)

#load data
mpi_regions = pd.read_csv("kiva_mpi_region_locations.csv")

# aggregate
loan_amount_by_country = kiva_loans.groupby('country')['loan_amount'].sum().reset_index()

# merge datasets
merged_data = mpi_regions.merge(loan_amount_by_country, on='country')

# load world boundries
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# merge the loan amount data with the world boundaries data
world_loans = world.merge(loan_amount_by_country, left_on='name', right_on='country')

# plot
fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

# use the 'viridis' colormap
norm = mcolors.Normalize(vmin=world_loans['loan_amount'].min(), vmax=world_loans['loan_amount'].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=cax)
cbar.ax.set_ylabel('Loan Amount', rotation=270, labelpad=20)

# set ticks and labels
ticks = np.linspace(world_loans['loan_amount'].min(), world_loans['loan_amount'].max(), 5)
cbar.set_ticks(ticks)

# round tick values 
cbar.set_ticklabels(['${:,.0f}'.format(round(tick, -3)) for tick in ticks])

world_loans.plot(ax=ax, column='loan_amount', cmap='viridis', edgecolor='black', linewidth=0.5, legend=False)
world.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
ax.set_title("Loan Distribution by Country", fontsize=16, fontweight='bold')
ax.set_xlabel("Longitude", fontsize=14, fontweight='bold')
ax.set_ylabel("Latitude", fontsize=14, fontweight='bold')
plt.show()
