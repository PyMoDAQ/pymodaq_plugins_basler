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

The plugin was tested using an acA640-120gm camera. It is compatible with PyMoDAQ 4.4.7.

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

