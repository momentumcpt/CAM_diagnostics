from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import warnings  # use to warn user about missing files.
import matplotlib as mpl
#Set non-X-window backend for matplotlib:                                                     
mpl.use('Agg')
#Now import pyplot:                                                                           
import matplotlib.pyplot as plt

def station_profiles(case_name, model_rgrid_loc, data_name, data_loc,
                     var_list, data_list, plot_location):

    """
    This script plots profiles of variables at specific locations.
    It currently does not compare against observations or baseline runs.

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
                        This is currently not used in this script.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
                        This is currently not used in this script.
    var_list         -> List of CAM output variables provided by "diag_var_list"
                        The can be CAM or CLUBB vars that are written to h0 files.
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".
    stations         -> List of Latitudes and Longitudes for the locations that 
                        profile plots will be created from.
    Notes:
        The script produces plots of only 3-D variables.
        Currently, the default behavior is to interpolate
        climo files to pressure levels, which requires the hybrid-sigma
        coefficients and surface pressure. That ASSUMES that the climo
        files are using native hybrid-sigma levels rather than being
        transformed to pressure levels.
    """

    print("  Generating station profile plots...")

    #Set input/output data path variables:
    #------------------------------------
    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_root     = Path(plot_location)
    plot_loc      = plot_root / '{}_vs_{}'.format(case_name, data_name)
    #-----------------------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Set plot file type:
    plot_type = 'png'

    # Temporary test location
    locations = [[30,90],[0,25]]

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)

    # probably want to do this one variable at a time:                                                  
    for var in var_list:

        #Notify user of variable being plotted:   
        print("\t \u231B profile plots for {}".format(var))

        #loop over different data sets to plot model against:                                           
        for data_src in data_list:

            # load data (observational) commparison files 
            oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))
            oclim_ds = _load_dataset(oclim_fils)

            # load re-gridded model files:
            mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))
            mclim_ds = _load_dataset(mclim_fils)

            # stop if data is invalid:
            if (oclim_ds is None) or (mclim_ds is None):
                warnings.warn(f"invalid data, skipping zonal mean plot of {var}")
                continue

            #Extract variable of interest
            odata = oclim_ds[var].squeeze()  # squeeze in case of degenerate dimensions 
            mdata = mclim_ds[var].squeeze()

            #                 
            # Seasonal Averages
            # Create new dictionaries:
            mseasons = dict()
            oseasons = dict()
            dseasons = dict() # hold the differences


            # Check the dimensions in data, profiles only available for 3D fields
            has_lat, has_lev = pf.zm_validate_dims(mdata)
            if not has_lev:
                print("{} has no lev (vertical) dimension... skipping profiles".format(var))
            else:
                #Loop over season dictionary:
                for s in seasons:
                    mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                    oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')

                    # Load time averaged profile at location
                    for loc in locations:
                        tlat = loc[0]
                        tlon = loc[1]
                        plot_name = plot_loc / "{}_{}_Profile_{}N{}E_Mean.{}".format(var,s,tlat,tlon,plot_type)
                        prof_mdata = mseasons[s].sel(lat=tlat,method="nearest").sel(lon=tlon,method="nearest")
                        prof_odata = oseasons[s].sel(lat=tlat,method="nearest").sel(lon=tlon,method="nearest")
                        #Remove old plot, if it already exists
                        if plot_name.is_file():
                            plot_name.unlink()

                        #Create New Plot
                        plot_prof_and_save(plot_name, prof_mdata, prof_odata, var, tlat, tlon)


    #Notify user that script has ended:
    print("  ...Station profile have been generated successfully.")


#
# Helpers
#
def _load_dataset(fils):
    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)

####### 
def plot_prof_and_save(plot_name, mdlfld, obsfld, var, tlat, tlon, **kwargs):
    """This plots a profile of mdlfld and obsfld at a location ."""

    plt_title = "Tot Mean {} at {}lon {}lat".format(var, tlon, tlat)

    #fig, (ax1,ax2) = plt.subplots(1,2)
    #fig.suptitle(plt_title)
    #ax1.plot(mdlfld,mdlfld['lev'])
    #ax1.invert_yaxis()
    #ax2.plot(obsfld,obsfld['lev'])
    #ax2.invert_yaxis()

    if 'lev' in mdlfld.dims:
        plt.plot(mdlfld,mdlfld['lev'], label="Test")
        plt.plot(obsfld,obsfld['lev'], label="Baseline")
    elif 'ilev' in mdlfld.dims:
        plt.plot(mdlfld,mdlfld['ilev'], label="Test")
        plt.plot(obsfld,obsfld['ilev'], label="Baseline")
    
    plt.legend()
    plt.title(plt_title)
    ax1 = plt.axes()
    ax1.invert_yaxis()

    #Write the figure to provided workspace/file:                                                              
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    #Close plots:                                                                                              
    plt.close()
