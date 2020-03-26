"""
Location of the "cam_diag" object, which
is used to store all relevant data and
info needed for generating CAM
diagnostics, including info
on the averaging, regridding, and
plotting methods themselves.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import sys
import os.path
import glob
from pathlib import Path
import subprocess

#Check if "PyYAML" is present in python path:
try:
    import yaml
except ImportError:
    print("PyYAML module does not exist in python path.")
    print("Please install module, e.g. 'pip install pyyaml'.")
    sys.exit(1)

#+++++++++++++++++++++++++++++
#Add Cam diagnostics 'scripts'
#directories to Python path
#+++++++++++++++++++++++++++++

#Determine local directory path:
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

#Add "scripts" directory to path:
_DIAG_SCRIPTS_PATH = os.path.join(_LOCAL_PATH,os.pardir,"scripts")

#Check that "scripts" directory actually exists:
if not os.path.isdir(_DIAG_SCRIPTS_PATH):
    #If not, then raise error:
    raise FileNotFoundError("'{}' directory not found. Has 'CamDiag.py' been moved?".format(_DIAG_SCRIPTS_PATH))

#Walk through all sub-directories in "scripts" directory:
for root, dirs, files in os.walk(_DIAG_SCRIPTS_PATH):
    #Add all sub-directories to python path:
    for dirname in dirs:
        sys.path.append(os.path.join(root,dirname))

#################
#Helper functions
#################

#++++++++++++++++++++++++++++++++
#Script message and exit function
#++++++++++++++++++++++++++++++++

def end_diag_script(msg):

    """
    Prints message, and then exits script.
    """

    print("\n")
    print(msg)
    sys.exit(1)

def read_namelist_obj(namelist_obj, varname):

    """
    Checks if variable/list/dictionary exists in
    namelist object,and if so returns it.
    """

    try:
        var = namelist_obj[varname]
    except:
       raise KeyError("'{}' not found in namelist file.  Please see 'example_namelist.yaml'.".format(varname))

    #return variable/list/dictionary:
    return var

######################################
#Main CAM diagnostics class (cam_diag)
######################################

class CamDiag:

    """
    Main CAM diagnostics object.

    This object is initalized using
    a CAM diagnostics namelist (YAML) file,
    which specifies various user inputs,
    including CAM history file names and
    locations, years being analyzed,
    types of averaging, regridding,
    and other post-processing options being
    used, and the type of plots that will
    be created.

    This object also contains various methods
    used to actually generate the plots and
    post-processed data.
    """

    def __init__(self, namelist_file):

        """
        Initalize CAM diagnostics object.
        """

        #Check that YAML file actually exists:
        if not os.path.exists(namelist_file):
            raise FileNotFoundError("'{}' file not found.".format(namelist_file))

        #Open YAML file:
        with open(namelist_file) as nfil:
            #Load YAML file:
            namelist = yaml.load(nfil, Loader=yaml.SafeLoader)

        #Add basic diagnostic info to object:
        self.__basic_info = read_namelist_obj(namelist, 'diag_basic_info')

        #Check if CAM climatology files will be calculated:
        try:
            if namelist['diag_cam_climo']['calc_cam_climo']:
                self.__cam_climo_info = read_namelist_obj(namelist, 'diag_cam_climo')
        except:
            raise KeyError("'calc_cam_climo' in 'diag_cam_climo' not found in namelist file.  Please see 'example_namelist.yaml'.")

        #Check if CAM baseline climatology files will be calculated:
        if not self.__basic_info['compare_obs']:
            try:
                if namelist['diag_cam_baseline_climo']['calc_bl_cam_climo']:
                    self.__cam_bl_climo_info = read_namelist_obj(namelist, 'diag_cam_baseline_climo')
            except:
                raise KeyError("'calc_bl_cam_climo' in 'diag_cam_baseline_climo' not found in namelist file.  Please see 'example_namelist.yaml'.")

        #Add averaging script name:
        self.__averaging_script = read_namelist_obj(namelist, 'averaging_script')

        #Add regridding script name:
        self.__regridding_script = read_namelist_obj(namelist, 'regridding_script')

        #Add plotting script names:
        self.__plot_scripts = read_namelist_obj(namelist, 'plotting_scripts')

        #Add CAM variable list:
        self.__diag_var_list = read_namelist_obj(namelist, 'diag_var_list')

    #########

    def create_time_series(self, baseline=False):

        """
        Generate time series versions of the CAM history file data.
        """

        #Check if baseline time-series files are being created:
        if baseline:
            #Then use the CAM baseline climo dictionary:
            cam_climo_dict = self.__cam_bl_climo_info
        else:
            #If not, then just extract the standard CAM climo dictionary:
            cam_climo_dict = self.__cam_climo_info

        #Check if climatologies are being calculated:
        if cam_climo_dict['calc_cam_climo']:

            #Extract case name:
            case_name = cam_climo_dict['cam_case_name']

            #Extract cam time series directory:
            ts_dir = cam_climo_dict['cam_ts_loc']

            #Extract cam variable list:
            var_list = self.__diag_var_list

            #Create path object for the CAM history file(s) location:
            starting_location = Path(cam_climo_dict['cam_hist_loc'])

            #Create ordered list of CAM history files:
            hist_files = sorted(list(starting_location.glob('*.cam.h0.*')))

            #Check that CAM history files exist.  If not then kill script:
            if not hist_files:
                msg = "No CAM history (h0) files found in {}.  Script is ending here."
                end_diag_script(msg)

            #Check if time series directory exists, and if not, then create it:
            if not os.path.isdir(ts_dir):
                print("{} not found, making new directory".format(ts_dir))
                os.mkdir(ts_dir)

            #Loop over CAM history variables:
            for var in var_list:

                #Create full path name:
                ts_outfil_str = ts_dir + os.sep + case_name + \
                              ".ncrcat."+var+".nc"

                #Check if files already exist in time series directory:
                ts_file_list = glob.glob(ts_outfil_str)

                #If files exist, then check if over-writing is allowed:
                if ts_file_list:
                    if not cam_climo_dict['cam_overwrite_ts']:
                        #If not, then simply skip this variable:
                        continue

                #Generate filename without suffix:
                first_in  = Path(hist_files[0])
                case_spec = first_in.stem

                #Notify user of new time series file:
                print("Creating new time series file for {}".format(var))

                #Run "ncrcat" command to generate time series file:
                cmd = ["ncrcat", "-4", "-h", "-v", var] + hist_files + ["-o", ts_outfil_str]
                subprocess.run(cmd)

        else:
            #If not, then notify user that time series generation is skipped.
            print("Climatology files are not being generated, so neither will time series files.")

    #########

    def create_plots(self):

        """
        Generate CAM diagnositc plots.
        """

        #Notify user that plot creation is starting:
        print("Creating CAM diagnostic plots.")

        #How to actually run plotting scripts:
        #for plot_script in self.__plot_scripts:
           #Add '()' to script name:
        #   plot_func = plot_script+'()'
        #   eval(plot_func)


###############
