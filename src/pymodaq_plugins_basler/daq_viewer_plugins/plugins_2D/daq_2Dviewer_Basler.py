import numpy as np
import os
import imageio as iio

from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters, params


# Suppress only NumPy RuntimeWarnings (bc of crosshair bug)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from pymodaq_plugins_basler.hardware.basler import BaslerCamera, TemperatureMonitor
from qtpy import QtWidgets, QtCore

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore


class DAQ_2DViewer_Basler(DAQ_Viewer_base):
    """Viewer for Basler cameras
    """
    controller: BaslerCamera
    live_mode_available = True

    # For Basler, this returns a list of user defined camera names
    camera_list = [cam.GetFriendlyName() for cam in BaslerCamera.list_cameras()]

    # Default place to store qsettings for this module
    settings_basler = QtCore.QSettings("PyMoDAQ", "Basler")

    # Update the params
    params = comon_parameters + [{'title': 'Camera List:', 'name': 'camera_list', 'type': 'list', 'value': '', 'limits': camera_list},
        {"title": "Device Info", "name": "device_info", "type": "group", "children": [
            {"title": "Device Model Name", "name": "DeviceModelName", "type": "str", "value": "", "readonly": True},
            {"title": "Device Serial Number", "name": "DeviceSerialNumber", "type": "str", "value": "", "readonly": True},
            {"title": "Device Version", "name": "DeviceVersion", "type": "str", "value": "", "readonly": True},
            {"title": "Device User ID", "name": "DeviceUserID", "type": "str", "value": ""}
        ]},
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2], 'default': 1},
            {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'readonly': True},
            {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'readonly': True},
        ]},
        ]

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: None
        self.user_id = None

        self.data_shape = None
        self.save_frame = False

    def init_controller(self) -> BaslerCamera:
        # Init camera 
        self.user_id = self.settings.param('camera_list').value()
        self.emit_status(ThreadCommand('Update_Status', [f"Trying to connect to {self.user_id}", 'log']))
        camera_list = BaslerCamera.list_cameras()
        for devInfo in camera_list:
            if devInfo.GetFriendlyName() == self.user_id:
                return BaslerCamera(info=devInfo, callback=self.emit_data_callback)
        self.emit_status(ThreadCommand('Update_Status', ["Camera not found", 'log']))
        raise ValueError(f"Camera with name {self.user_id} not found anymore.")

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        # Initialize camera class
        self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())
        
        # Setup continuous acquisition & allow adjustable frame rate
        self.controller.setup_acquisition()

        # Connect camera lost event
        self.controller.configurationEventHandler.signals.cameraRemoved.connect(self.camera_lost)
        
        # Update the UI with available and current camera parameters
        self.add_attributes_to_settings()
        self.update_params_ui()
        for param in self.settings.children():
            if param.name() == 'device_info':
                continue
            param.sigValueChanged.emit(param, param.value())
            if param.hasChildren():
                for child in param.children():
                    child.sigValueChanged.emit(child, child.value())

        # Update image parameters
        (x0, xend, y0, yend, xbin, ybin) = self.controller.get_roi()
        height = xend - x0
        width = yend - y0
        self.settings.child('roi', 'binning').setValue(xbin)
        self.settings.child('roi', 'width').setValue(width)
        self.settings.child('roi', 'height').setValue(height)
                
        self._prepare_view()
        info = "Initialized camera"
        print(f"{self.user_id} camera initialized successfully")
        self.emit_status(ThreadCommand('Update_Status', [f"{self.user_id} camera initialized successfully"]))
        initialized = True
        return info, initialized

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        name = param.name()
        value = param.value()

        if name == "camera_list":
            if self.controller != None:
                self.close()
            self.ini_detector()

        if name == "device_state_save":
            self.controller.save_device_state()
            param = self.settings.child('device_state', 'device_state_save')
            param.setValue(False)
            param.sigValueChanged.emit(param, False)
            return
        
        if name == "device_state_load":
            self.controller.stop_grabbing()
            self.controller.load_device_state()
            # Reinitialize what is needed
            self.controller.setup_acquisition()
            # Update the UI with available and current camera parameters
            self.add_attributes_to_settings()
            self.update_params_ui()
            for param in self.settings.children():
                param.sigValueChanged.emit(param, param.value())
                if param.hasChildren():
                    for child in param.children():
                        child.sigValueChanged.emit(child, child.value())
            self._prepare_view()
            self.controller.start_grabbing(self.settings.param('AcquisitionFrameRateAbs').value())
            self.emit_status(ThreadCommand('Update_Status', [f"Device state loaded"]))
            return
        
        if name == 'PixelFormat':
            self.controller.stop_grabbing()
            self.controller.camera.PixelFormat.SetValue(value)
            self._prepare_view()
            self.controller.start_grabbing(self.settings.param('AcquisitionFrameRateAbs').value())
            return
        
        if name == 'TriggerSave':
            if not self.settings.child('trigger', 'TriggerMode').value():
                print("Trigger mode is not active ! Start triggering first !")
                self.emit_status(ThreadCommand('Update_Status', ["Trigger mode is not active ! Start triggering first !"]))
                param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                param.setValue(False) # Turn off save on trigger if triggering is off
                param.sigValueChanged.emit(param, False) 
                return
            if value:
                self.save_frame = True
                return
            else:
                self.save_frame = False
                return
    
        if name in self.controller.attribute_names:
            # Special cases
            if 'ExposureTime' in name:
                value = int(value * 1e3)
            if 'Gain' in name and 'Auto' not in name:
                value = int(value)
            if name == "DeviceUserID":
                self.user_id = value
                self.controller.camera.DeviceUserID.SetValue(value)
                # Update the camera list to account for name change 
                camera_list = [cam.GetFriendlyName() for cam in BaslerCamera.list_cameras()]
                param = self.settings.param('camera_list')
                param.setLimits(camera_list)
                param.sigLimitsChanged.emit(param, camera_list)
                return
            if name == 'TriggerMode':
                camera_attr = getattr(self.controller.camera, name)
                if value:
                    camera_attr.SetIntValue(1)
                else:
                    self.save_frame = False
                    camera_attr.SetIntValue(0)
                    param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                    param.setValue(False) # Turn off save on trigger if we turn off triggering
                    param.sigValueChanged.emit(param, False)
                return
            if name == 'GainAuto':
                camera_attr = getattr(self.controller.camera, name)
                if value:
                    camera_attr.SetIntValue(2)
                else:
                    camera_attr.SetIntValue(0)
                return
            if name == 'ExposureAuto':
                camera_attr = getattr(self.controller.camera, name)
                if value:
                    camera_attr.SetIntValue(2)
                else:
                    camera_attr.SetIntValue(0)
                return
            # we only need to reference these, nothing to do with the cam
            if name == 'TriggerSaveLocation':
                return
            if name == 'TriggerSaveIndex':
                return
            if name == 'Filetype':
                return
            if name == 'Prefix':
                return
            if name == 'TemperatureMonitor':
                if value:
                    # Start thread for camera temp. monitoring
                    self.start_temperature_monitoring()
                else:
                    # Stop background threads
                    self.stop_temp_monitoring()
                return

            # All the rest, just do :
            camera_attr = getattr(self.controller.camera, name)
            camera_attr.SetValue(value)

        if name == "update_roi":
            if value:  # Switching on ROI

                # We handle ROI and binning separately for clarity
                (old_x, _, old_y, _, xbin, ybin) = self.controller.get_roi()  # Get current binning
                y0, x0 = self.roi_info.origin.coordinates
                height, width = self.roi_info.size.coordinates

                # Values need to be rescaled by binning factor and shifted by current x0,y0 to be correct.
                new_x = (old_x + x0) * xbin
                new_y = (old_y + y0) * xbin
                new_width = width * ybin
                new_height = height * ybin
                
                new_roi = (new_x, new_width, xbin, new_y, new_height, ybin)
                self.update_rois(new_roi)
                param.setValue(False)
                param.sigValueChanged.emit(param, False)
        elif name == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child('roi', 'binning').value()
            ybin = self.settings.child('roi', 'binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)
        elif name == "clear_roi":
            if value:  # Switching on ROI
                wdet, hdet = self.controller.get_detector_size()
                self.settings.child('roi', 'binning').setValue(1)

                new_roi = (0, wdet, 1, 0, hdet, 1)
                self.update_rois(new_roi)
                param.setValue(False)
                param.sigValueChanged.emit(param, False)


    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""
 
        (hstart, hend, vstart, vend, *binning) = self.controller.get_roi()
        try:
           xbin, ybin = binning
        except ValueError:  # some Pylablib `get_roi` do return just four values instead of six
           xbin = ybin = 1
        height = hend - hstart
        width = vend - vstart
 
        self.settings.child('roi', 'width').setValue(width)
        self.settings.child('roi', 'height').setValue(height)

        mock_data = np.zeros((height, width))

        self.x_axis = Axis(label='Pixels', data=np.linspace(1, width, width), index=1)

        if width != 1 and height != 1:
            data_shape = 'Data2D'
            self.y_axis = Axis(label='Pixels', data=np.linspace(1, height, height), index=0)
            self.axes = [self.y_axis, self.x_axis]
        else:
            data_shape = 'Data1D'
            self.axes = [self.x_axis]

        if data_shape != self.data_shape:
            self.data_shape = data_shape
            self.dte_signal_temp.emit(
                DataToExport(f'{self.user_id}',
                            data=[DataFromPlugins(name=f'{self.user_id}',
                                                    data=[np.squeeze(mock_data)],
                                                    dim=self.data_shape,
                                                    labels=[f'{self.user_id}_{self.data_shape}'],
                                                    axes=self.axes)]))

            QtWidgets.QApplication.processEvents()

    def update_rois(self, new_roi):
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.controller.get_roi():
            # self.controller.set_attribute_value("ROIs",[new_roi])
            self.controller.set_roi(hstart=new_x,
                                    hend=new_x + new_width,
                                    vstart=new_y,
                                    vend=new_y + new_height,
                                    hbin=new_xbinning,
                                    vbin=new_ybinning)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.controller.clear_acquisition()
            self.controller.setup_acquisition()
            # Finally, prepare view for displaying the new data
            self._prepare_view()

    def grab_data(self, Naverage: int = 1, live: bool = False, **kwargs) -> None:
        try:
            self._prepare_view()
            if "Acquisition Frame Rate" in self.controller.attributes:
                frame_rate = self.settings.param('AcquisitionFrameRateAbs').value()
            else:
                frame_rate = None
            if live:
                self.controller.start_grabbing(frame_rate)
            else:
                self.controller.start_grabbing(frame_rate)
                while not self.controller.imageEventHandler.frame_ready:
                    pass # do nothing until a frame is ready
                self.controller.stop_grabbing()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))


    def emit_data_callback(self, frame_data: dict) -> None:
        frame = frame_data['frame']
        shape = frame.shape
        # First emit data to the GUI
        dte = DataToExport(f'{self.user_id}', data=[DataFromPlugins(
            name=f'{self.user_id}',
            data=[np.squeeze(frame)],
            dim=self.data_shape,
            labels=[f'{self.user_id}_{self.data_shape}'],
            axes=self.axes)])
        self.dte_signal.emit(dte)

        # Now, handle data saving with filepath given by user in trigger save settings
        if self.save_frame:
            index = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveIndex')
            filetype = self.settings.child('trigger', 'TriggerSaveOptions', 'Filetype').value()
            filepath = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveLocation').value()
            prefix = self.settings.child('trigger', 'TriggerSaveOptions', 'Prefix').value()
            if not filepath:
                filepath = os.path.join(os.path.expanduser('~'), 'Downloads')
            filename = f"{prefix}{index.value()}.{filetype}"
            index.setValue(index.value()+1)
            index.sigValueChanged.emit(index, index.value())
            if not filename.endswith(f".{filetype}"):
                filename += f".{filetype}"
            full_path = os.path.join(filepath, f"{filename}")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            try:
                iio.imwrite(full_path, frame)
            except Exception as e:
                print(f"Failed to save image: {e}")
                self.emit_status(ThreadCommand('Update_Status', [f"Failed to save image: {e}"]))

        # Prepare for next frame
        self.controller.imageEventHandler.frame_ready = False

        # Update exposure and gain in GUI if set to auto
        exposure_name, gain_name, raw_gain = self.controller.check_attribute_names()
        if self.settings.child('exposure', 'ExposureAuto').value():
            camera_attr = getattr(self.controller.camera, exposure_name)
            value = camera_attr.GetValue()
            value = int(value * 1e-3)
            param = self.settings.child('exposure', exposure_name)
            param.setValue(value)
            param.sigValueChanged.emit(param, param.value())
        if self.settings.child('gain', 'GainAuto').value():
            camera_attr = getattr(self.controller.camera, gain_name)
            value = camera_attr.GetValue()
            param = self.settings.child('gain', gain_name)
            param.setValue(value)
            param.sigValueChanged.emit(param, param.value())

    def stop(self):
        self.controller.camera.StopGrabbing()
        return ''
    
    def close(self):
        """Terminate the communication protocol"""
        self.controller.attributes = None
        self.controller.close()
            
        # Stop any background threads
        try:
            self.stop_temp_monitoring()
        except Exception:
            pass # no temp settings

        # Just set these to false if camera disconnected for clean GUI
        try:
            param = self.settings.child('trigger', 'TriggerMode')
            param.setValue(False) # Turn off save on trigger if triggering is off
            param.sigValueChanged.emit(param, False)
            param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
            param.setValue(False) # Turn off save on trigger if triggering is off
            param.sigValueChanged.emit(param, False) 
        except Exception:
            pass # no trigger settings

        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""
        print(f"{self.user_id} communication terminated successfully")
        self.emit_status(ThreadCommand('Update_Status', [f"{self.user_id} communication terminated successfully"]))
    
    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        self.crosshair_info = crosshair_info
        # Adding a small delay improves performance 
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.processEvents)

    def camera_lost(self):
        self.close()
        print(f"Lost connection to {self.user_id}")
        self.emit_status(ThreadCommand('Update_Status', [f"Lost connection to {self.user_id}"]))

    def start_temperature_monitoring(self):
        self.temp_thread = QtCore.QThread()
        self.temp_worker = TemperatureMonitor(self.controller.camera)

        self.temp_worker.moveToThread(self.temp_thread)

        self.temp_thread.started.connect(self.temp_worker.run)
        self.temp_worker.temperature_updated.connect(self.on_temperature_update)
        self.temp_worker.finished.connect(self.temp_thread.quit)
        self.temp_worker.finished.connect(self.temp_worker.deleteLater)
        self.temp_thread.finished.connect(self.temp_thread.deleteLater)

        self.temp_thread.start()

    def stop_temp_monitoring(self):
        if hasattr(self, 'temp_worker') and self.temp_worker is not None:
            self.temp_worker.stop()
            self.temp_worker = None
        if hasattr(self, 'temp_thread') and self.temp_thread is not None:
            try:
                self.temp_thread.quit()
                self.temp_thread.wait()
            except RuntimeError:
                pass  # Already deleted
            self.temp_thread = None
        # Make sure temp. monitoring param is false in GUI
        param = self.settings.child('temperature', 'TemperatureMonitor')
        param.setValue(False)
        param.sigValueChanged.emit(param, param.value())

    def on_temperature_update(self, temp: float):
        param = self.settings.child('temperature', 'TemperatureAbs')
        param.setValue(temp)
        param.sigValueChanged.emit(param, temp)
        # TODO maybe close device here if temperature is too high, and allow user to set this threshold ?
        if temp > 60:
            self.emit_status(ThreadCommand('Update_Status', [f"WARNING: {self.user_id} camera is hot !!"]))


    # This will add self.attributes, which is derived from the model config file, to self.settings
    def add_attributes_to_settings(self):
        existing_group_names = {child.name() for child in self.settings.children()}

        for attr in self.controller.attributes:
            attr_name = attr['name']
            if attr.get('type') == 'group':
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)
                else:
                    group_param = self.settings.child(attr_name)

                    existing_children = {child.name(): child for child in group_param.children()}

                    expected_children = attr.get('children', [])
                    for expected in expected_children:
                        expected_name = expected['name']
                        if expected_name not in existing_children:
                            for old_name, old_child in existing_children.items():
                                if old_child.opts.get('title') == expected.get('title') and old_name != expected_name:
                                    self.settings.child(attr_name, old_name).show(False)
                                    break

                            group_param.addChild(expected)
            else:
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)
        
    # This will ensure that the UI shows the current camera parameters values and limits
    def update_params_ui(self):

        # Common syntax for any camera model
        param = self.settings.child('device_info','DeviceModelName').setValue(self.controller.model_name)
        self.settings.child('device_info','DeviceSerialNumber').setValue(self.controller.device_info.GetSerialNumber())
        self.settings.child('device_info','DeviceVersion').setValue(self.controller.device_info.GetDeviceVersion())
        self.settings.child('device_info','DeviceUserID').setValue(self.controller.device_info.GetFriendlyName())


        for param in self.controller.attributes:
            param_type = param['type']
            param_name = param['name']
            
            # Already handled
            if param_name == "device_info":
                continue
            if param_name == "device_state":
                continue
            if param_name == "temperature":
                continue

            if param_type == 'group':
                # Recurse over children in groups
                for child in param['children']:
                    child_name = child['name']
                    child_type = child['type']
                    # Special case: skip these
                    if child_name == 'TriggerSaveOptions':
                        continue

                    camera_attr = getattr(self.controller.camera, child_name)

                    try:
                        if child_type in ['float', 'slide', 'int', 'str']:
                            value = camera_attr.GetValue()
                        elif child_type == 'led_push':
                            if child_name != 'GammaEnable':
                                value = bool(camera_attr.GetIntValue())
                            else:
                                value = camera_attr.GetValue()
                        else:
                            continue  # Unsupported type, skip

                        # Special case: if parameter is related to ExposureTime, convert to ms from us
                        if 'Exposure' in child_name and 'Auto' not in child_name:
                            value *= 1e-3

                        # Set the value
                        self.settings.child(param_name, child_name).setValue(value)

                        # Set limits if defined
                        if 'limits' in child and child_type in ['float', 'slide', 'int'] and not child.get('readonly', False):
                            try:
                                min_limit = camera_attr.GetMin()
                                max_limit = camera_attr.GetMax()

                                if 'Exposure' in child_name and 'Auto' not in child_name:
                                    min_limit *= 1e-3
                                    max_limit *= 1e-3

                                self.settings.child(param_name, child_name).setLimits([min_limit, max_limit])
                            except Exception:
                                pass

                    except Exception:
                        pass
            else:
                camera_attr = getattr(self.controller.camera, param_name)
                try:
                    if param_type in ['float', 'slide', 'int', 'str']:
                        value = camera_attr.GetValue()
                    elif param_type == 'led_push':
                        if param_name != 'GammaEnable':
                            value = bool(camera_attr.GetIntValue())
                        else:
                            value = camera_attr.GetValue()
                    else:
                        continue  # Unsupported type, skip

                    # Special case: if parameter is related to ExposureTime, convert to ms from us
                    if 'Exposure' in param_name and 'Auto' not in param_name:
                        value *= 1e-3

                    # Set the value
                    self.settings.param(param_name).setValue(value)

                    if 'limits' in param and param_type in ['float', 'slide', 'int'] and not param.get('readonly', False):
                        try:
                            min_limit = camera_attr.GetMin()
                            max_limit = camera_attr.GetMax()


                            if 'Exposure' in param_name and 'Auto' not in param_name:
                                min_limit *= 1e-3
                                max_limit *= 1e-3

                            self.settings.param(param_name).setLimits([min_limit, max_limit])

                        except Exception:
                            pass

                except Exception:
                    pass


if __name__ == '__main__':
    main(__file__, init=False)