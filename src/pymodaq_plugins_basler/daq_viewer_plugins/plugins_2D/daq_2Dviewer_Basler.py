import numpy as np
import os
import imageio as iio
import h5py

from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters

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

    # Update the params
    params = comon_parameters + [{'title': 'Camera List:', 'name': 'camera_list', 'type': 'list', 'value': '', 'limits': camera_list},
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2], 'default': 1},
            {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'readonly': True},
            {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'readonly': True},
        ]}]

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: None
        self.user_id = None

        self.data_shape = None
        self.save_frame_local = False
        self.frame_saved = False

        # For LECO operation
        self.metadata = None
        self.save_frame_leco = False

    def init_controller(self) -> BaslerCamera:
        # Init camera 
        self.user_id = self.settings.param('camera_list').value()
        self.emit_status(ThreadCommand('Update_Status', [f"Trying to connect to {self.user_id}", 'log']))
        device_info = BaslerCamera.list_cameras()
        for devInfo in device_info:
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

        # Connect camera lost event
        self.controller.configurationEventHandler.signals.cameraRemoved.connect(self.camera_lost)

        # Setup continuous acquisition & allow adjustable frame rate
        self.controller.setup_acquisition()

        # Start thread for camera temp. monitoring
        self.start_temperature_monitoring()

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
        try:
            camera_attr = getattr(self.controller.camera, name)
        except AttributeError:
            pass

        if name == "camera_list":
            if self.controller != None:
                self.close()
            self.ini_detector()

        if name == "device_state_save":
            self.controller.camera.device_save_state_to_file(self.controller.default_device_state_path)
            param = self.settings.child('device_state', 'device_state_save')
            param.setValue(False)
            param.sigValueChanged.emit(param, False) 
            return
        if name == "device_state_load":
            filepath = self.settings.child('device_state', 'device_state_to_load').value()
            self.controller.stop_grabbing()
            self.controller.load_device_state(filepath)
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
            self.emit_status(ThreadCommand('Update_Status', [f"Device state loaded from {filepath}"]))
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
                self.save_frame_local = True
                return
            else:
                self.save_frame_local = False
                return
        if name == 'PixelFormat':
            self.controller.stop_grabbing()
            self.controller.camera.PixelFormat.SetValue(value)
            self._prepare_view()
            self.controller.start_grabbing(self.settings.param('AcquisitionFrameRateAbs').value())
            return
    
        if name in self.controller.attribute_names:
            # Special cases
            if 'ExposureTime' in name:
                value = int(value * 1e3)
            if 'Gain' in name and 'Auto' not in name:
                value = int(value)
            if name == "DeviceUserID":
                self.user_id = value
                #TODO: Fix this so camera name change persists
                #self.controller.camera.DeviceInfo.SetFriendlyName(value)
                # Update the camera list to account for name change 
                #camera_list = [cam.GetFriendlyName() for cam in BaslerCamera.list_cameras()]
                #param = self.settings.param('camera_list')
                #param.setLimits(camera_list)
                #param.sigLimitsChanged.emit(param, camera_list)
                return
            if name == 'TriggerMode':
                if value:
                    camera_attr.SetIntValue(1)
                else:
                    camera_attr.SetIntValue(0)
                    param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                    param.setValue(False) # Turn off save on trigger if we turn off triggering
                    param.sigValueChanged.emit(param, False)
                    self.save_frame_local = False
                return
            if name == 'GainAuto':
                if value:
                    camera_attr.SetIntValue(1)
                else:
                    camera_attr.SetIntValue(0)
                return
            if name == 'ExposureAuto':
                if value:
                    camera_attr.SetIntValue(1)
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
            # All the rest, just do :
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

        # for self.user_id use camera.SetFriendlyName()

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
            if live:
                self.controller.start_grabbing(self.settings.param('AcquisitionFrameRateAbs').value())
            else:
                self.controller.start_grabbing(self.settings.param('AcquisitionFrameRateAbs').value())
                while not self.controller.imageEventHandler.frame_ready:
                    pass
                self.controller.stop_grabbing()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))


    def emit_data_callback(self, frame) -> None:
        if not self.save_frame_local and not self.save_frame_leco:
            dte = DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                axes=self.axes)])
        elif self.save_frame_local and not self.save_frame_leco:
            self.frame_saved = False
            dte = DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                do_save=True,
                axes=self.axes)])
            index = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveIndex')
            filepath = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveLocation').value()
            prefix = self.settings.child('trigger', 'TriggerSaveOptions', 'Prefix').value()
            filetype = self.settings.child('trigger', 'TriggerSaveOptions', 'Filetype').value()
            ulid = None
            if filetype == 'h5':
                if self.metadata is not None:
                    filepath = self.metadata['filepath']
                    filename = self.metadata['filename']
                    ulid = self.metadata['ulid']
                    with h5py.File(os.path.join(filepath, filename), 'w') as f:
                        f.create_dataset(filename, data=frame)
                        f.attrs['ulid'] = ulid
                        f.attrs['user_id'] = self.user_id
                    self.metadata = None
            else:
                if not filepath:
                    filepath = os.path.join(os.path.expanduser('~'), 'Downloads', f"{prefix}{index.value()}.{filetype}")
                else:
                    filepath = os.path.join(filepath, f"{prefix}{index.value()}.{filetype}")
                iio.imwrite(filepath, frame)
                index.setValue(index.value()+1)
                index.sigValueChanged.emit(index, index.value())
            self.frame_saved = True
        elif self.save_frame_leco:
            self.frame_saved = False
            dte = DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                do_save=True,
                axes=self.axes)])
            if self.metadata is not None:
                filepath = self.metadata['filepath']
                filename = self.metadata['filename']
                ulid = self.metadata['ulid']
                with h5py.File(os.path.join(filepath, filename), 'w') as f:
                    f.create_dataset(filename, data=frame)
                    f.attrs['ulid'] = ulid
                    f.attrs['user_id'] = self.user_id
                self.frame_saved = True
                self.metadata = None
        self.dte_signal.emit(dte)
        self.controller.imageEventHandler.frame_ready = False

    def stop(self):
        self.controller.camera.StopGrabbing()
        return ''
    
    def close(self):
        """Terminate the communication protocol"""
        self.controller.attributes = None
        self.controller.close()
            
        # Stop any background threads
        if hasattr(self, 'temp_worker'):
            self.temp_worker.stop()
        if hasattr(self, 'temp_thread'):
            self.temp_thread.quit()
            self.temp_thread.wait()

        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""
        print(f"{self.user_id} communication terminated successfully")
        self.emit_status(ThreadCommand('Update_Status', [f"{self.user_id} communication terminated successfully"]))
    
    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        sleep_ms = 150
        self.crosshair_info = crosshair_info
        QtCore.QTimer.singleShot(sleep_ms, QtWidgets.QApplication.processEvents)

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

    def on_temperature_update(self, temp: float):
        param = self.settings.child('misc', 'TemperatureAbs')
        param.setValue(temp)
        param.sigValueChanged.emit(param, temp)
        if temp > 60:
            self.emit_status(ThreadCommand('Update_Status', [f"WARNING: {self.user_id} camera is too hot !!"]))


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

            if param_type == 'group':
                # Recurse over children in groups
                for child in param['children']:
                    child_name = child['name']
                    child_type = child['type']
                    # Special case: skip these
                    if child_name == 'TriggerSaveOptions':
                        continue
                    if child_name == 'TemperatureAbs':
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
                        if 'ExposureTime' in child_name:
                            value *= 1e-3

                        # Set the value
                        self.settings.child(param_name, child_name).setValue(value)

                        # Set limits if defined
                        if 'limits' in child and child_type in ['float', 'slide', 'int'] and not child.get('readonly', False):
                            try:
                                min_limit = camera_attr.GetMin()
                                max_limit = camera_attr.GetMax()

                                if 'ExposureTime' in child_name:
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
                    if 'ExposureTime' in param_name:
                        value *= 1e-3

                    # Set the value
                    self.settings.param(param_name).setValue(value)

                    if 'limits' in param and param_type in ['float', 'slide', 'int'] and not param.get('readonly', False):
                        try:
                            min_limit = camera_attr.GetMin()
                            max_limit = camera_attr.GetMax()


                            if param_name == 'ExposureTime':
                                min_limit *= 1e-3
                                max_limit *= 1e-3

                            self.settings.param(param_name).setLimits([min_limit, max_limit])

                        except Exception:
                            pass

                except Exception:
                    pass


if __name__ == '__main__':
    main(__file__, init=False)