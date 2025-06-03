pymodaq_plugins_basler
######################

.. the following must be adapted to your developed package, links to pypi, github  description...

.. image:: https://img.shields.io/pypi/v/pymodaq-plugins-basler.svg
   :target: https://pypi.org/project/pymodaq-plugins-basler/
   :alt: Latest Version

.. image:: https://github.com/BenediktBurger/pymodaq_plugins_basler/workflows/Upload%20Python%20Package/badge.svg
   :target: https://github.com/BenediktBurger/pymodaq_plugins_basler
   :alt: Publication Status

.. image:: https://github.com/BenediktBurger/pymodaq_plugins_basler/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/BenediktBurger/pymodaq_plugins_basler/actions/workflows/Test.yml

Set of PyMoDAQ plugins for cameras by Basler, using the pypylon library. It handles basic camera functionalities (gain, exposure, ROI).
The data is emitted together with spatial axes corresponding either to pixels or to real-world units (um). The pixel size of different camera model is hardcoded in the hardware/basler.py file.
If the camera model is not specified, the pixel size is set to 1 um and can be changed manually by the user in the interface.

The plugin was tested using an acA640-120gm and acA1920-40gm camera. It is compatible with PyMoDAQ version greater than 4.4.7.

Config files are needed for different camera models. Example for the acA1920-40gm camera is given in the resources directory. 
The name of the config file should be config_<model_name> where <model_name> is the output of tlFactory.EnumerateDevices()[camera_index].GetModelName(). 
The module will look for this file in the ProgramData/.pymodaq/resources folder in Windows, and /etc/.pymodaq in Linux.
For most camera models, this config file will not need to be modified very much.
Simply rename the 'name' field in the different parameters to match the name that the camera uses with the pypylon API, i.e. GainRaw vs. GainAbs.
If the user wishes, any parameters can be removed as wanted/needed.
Adding new parameters is possible as of now, but only if it is a typical numeric parameter. 
Otherwise, the user will have to modify the code in the update_params_ui and commit_settings methods in the daq_2Dviewer_Basler.py file to handle these cases.
Other than parameters of type 'list', all parameters will automatically have their value and limits set from the camera upon initialization, so the exact values of these in the json is not important.

Authors
=======

* Benedikt Burger
* Romain Geneaux


Instruments
===========

Below is the list of instruments included in this plugin

Actuators
+++++++++

Viewer0D
++++++++

Viewer1D
++++++++

Viewer2D
++++++++

* **Basler**: control of Basler cameras


PID Models
==========


Extensions
==========


Installation instructions
=========================

* You need the manufacturer's driver `Pylon <https://www.baslerweb.com/pylon>`_ for the cameras.

