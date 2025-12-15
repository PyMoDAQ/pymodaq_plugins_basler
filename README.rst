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

Set of PyMoDAQ plugins for cameras by Basler, using the pypylon library. It handles basic camera functionalities (gain, exposure, ROI) and more advanced functionalities can be added by modifying a config file described below.
The data is emitted together with spatial axes corresponding either to pixels or to real-world units (um). The pixel size of different camera model is hardcoded in the hardware/basler.py file.
If the camera model is not specified, the pixel size is set to 1 um and can be changed manually by the user in the interface.

The plugin has been tested using acA640-120gm, acA1920-40gm, and puA1600-60um camera models. It is compatible with PyMoDAQ version greater than 4.4.7.

Config files are needed for different camera models. Examples for the acA1920-40gm and puA1600-60um camera models are given in the resources directory. 
The name of the config file should be config_<model_name> where <model_name> is the output of tlFactory.EnumerateDevices()[camera_index].GetModelName(). 
The module will look for this file in the ProgramData/.pymodaq folder in Windows and /etc/.pymodaq folder in Linux and if not found, a default config file can be created upon camera initialization.
The config file is a JSON which is expected to contain entries according to the ParameterTree structure of PyMoDAQ plugin parameters. The name of an entry is either a Group Parameter or the underlying camera feature, which must have the name that is used by the API, i.e. ExposureTime or GainRaw.

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
* Here are `complete installation instructions <https://pymodaq.cnrs.fr/en/latest/lab_story_folder/basler.html>`_.

