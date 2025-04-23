import numpy as np

from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters

# Suppress only NumPy RuntimeWarnings (bc of crosshair bug)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from pymodaq_plugins_basler.hardware.basler import BaslerCamera
from PyQt6.QtCore import pyqtSignal
from qtpy import QtWidgets, QtCore


class DAQ_2DViewer_Basler(DAQ_Viewer_base):
    """Viewer for Basler cameras
    """
    controller: BaslerCamera
    live_mode_available = True

    # For Basler, this returns a list of user defined camera names
    camera_list = [cam.GetUserDefinedName() for cam in BaslerCamera.list_cameras()]

    # Update the params
    params = comon_parameters + [
        {'title': 'Camera Identifiers', 'name': 'ID', 'type': 'group', 'children': [
            {'title': 'Camera List:', 'name': 'camera_list', 'type': 'list', 'value': '', 'limits': camera_list},
            {'title': 'Camera Model:', 'name': 'camera_model', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Camera Serial Number:', 'name': 'camera_serial', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Camera User ID:', 'name': 'camera_user_id', 'type': 'str', 'value': ''}
        ]},
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2], 'default': 1},
            {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'readonly': True},
            {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'readonly': True},

        ]},
        {'title': 'Exposure', 'name': 'exposure', 'type': 'group', 'children': [
            {'title': 'Auto Exposure', 'name': 'exposure_auto', 'type': 'led_push', 'value': 0, 'default': 0, 'limits': [0, 2]},
            {'title': 'Exposure Time (us)', 'name': 'exposure_time', 'type': 'int', 'value': 100, 'default': 100, 'limits': [0, 1]}
        ]},
        {'title': 'Gain', 'name': 'gain', 'type': 'group', 'children': [
            {'title': 'Auto Gain', 'name': 'gain_auto', 'type': 'led_push', 'value': 0, 'default': 0, 'limits': [0, 2]},
            {'title': 'Value', 'name': 'gain_value', 'type': 'int', 'value': 1, 'default': 1, 'limits': [0, 1]}
        ]},
        {'title': 'Frame Rate', 'name': 'frame_rate', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]},
        {'title': 'Gamma', 'name': 'gamma', 'type': 'group', 'children': [
            {'title': 'Gamma Enable', 'name': 'gamma_enable', 'type': 'led_push', 'value': 0, 'default': 0, 'limits': [0, 1]},
            {'title': 'Value', 'name': 'gamma_value', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]}
        ]},
        {'title': 'Inter-Packet Delay', 'name': 'gevscpd', 'type': 'int', 'value': 1, 'default': 1, 'limits': [0, 1]},
        {'title': 'Device Temperature', 'name': 'temp', 'type': 'int', 'value': 1, 'readonly': True}
    ]

    callback_signal = pyqtSignal()
    roi_info = None

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: None
        self.callback_thread = None
        self.user_id = None

        self.x_axis = None
        self.y_axis = None
        self.axes = None
        self.data_shape = None

    def init_controller(self) -> BaslerCamera:
        # Define the camera controller.
        # Use any argument necessary (serial_number, camera index, etc.) depending on the camera

        # Init camera with currently selected user id name
        self.user_id = self.settings.child('ID', 'camera_list').value()
        self.emit_status(ThreadCommand('Update_Status', [f"Trying to connect to {self.user_id}", 'log']))
        camera_list = BaslerCamera.list_cameras()
        for cam in camera_list:
            if cam.GetUserDefinedName() == self.user_id:
                name = cam.GetFullName()
                return BaslerCamera(name=name, callback=self.emit_data_callback)
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
        
        devInfo = self.controller.get_device_info()

        self.settings.child('ID', 'camera_model').setValue(devInfo[1])
        self.settings.child('ID', 'camera_serial').setValue(devInfo[2])
        self.settings.child('ID', 'camera_user_id').setValue(self.user_id)

        # Setup continuous acquisition & allow adjustable frame rate
        self.controller.setup_acquisition()
        self.controller.camera.AcquisitionFrameRateEnable = True

        # Check if pixel length is known
        if self.controller.pixel_length is None:
            self.emit_status(ThreadCommand('Update_Status', [(f"No pixel length known for camera model '{self.controller.model_name}', defaulting to user-chosen one"), 'log']))
            self.settings.child('pixel_length').show()

        try:
            param = getattr(self.controller.camera, self.controller.exposure_auto)
            self.settings.child('exposure', 'exposure_auto').setValue(param.GetIntValue())
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.exposure_time)
            self.settings.child('exposure', 'exposure_time').setValue(param.GetValue())
            self.settings.child('exposure', 'exposure_time').setDefault(param.GetValue())
            self.settings.child('exposure', 'exposure_time').setLimits([param.GetMin(), param.GetMax()])
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.gain_auto)
            self.settings.child('gain', 'gain_auto').setValue(param.GetIntValue())
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.gain_value)
            self.settings.child('gain', 'gain_value').setValue(param.GetValue())
            self.settings.child('gain', 'gain_value').setDefault(param.GetValue())
            self.settings.child('gain', 'gain_value').setLimits([param.GetMin(), param.GetMax()])
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.frame_rate)
            self.settings.param('frame_rate').setValue(param.GetMax())
            self.settings.param('frame_rate').setDefault(param.GetMax())
            self.settings.param('frame_rate').setLimits([param.GetMin(), param.GetMax()])
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.gamma_enable)
            self.settings.child('gamma', 'gamma_enable').setValue(param.GetIntValue())
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.gamma_value)
            self.settings.child('gamma', 'gamma_value').setValue(param.GetValue())
            self.settings.child('gamma', 'gamma_value').setDefault(param.GetValue())
            self.settings.child('gamma', 'gamma_value').setLimits([param.GetMin(), param.GetMax()])
        except Exception:
            pass
        try:
            param = getattr(self.controller.camera, self.controller.gevscpd)
            self.settings.param('gevscpd').setValue(self.controller.camera.GevSCPD.GetValue())
            self.settings.param('gevscpd').setDefault(self.controller.camera.GevSCPD.GetValue())
            self.settings.param('gevscpd').setLimits([self.controller.camera.GevSCPD.GetMin(), self.controller.camera.GevSCPD.GetMax()])
        except Exception:
            pass
        try:
            self.settings.param('temp').setValue(self.controller.camera.TemperatureAbs.GetValue())
        except Exception:
            pass

        # Update image parameters
        (x0, xend, y0, yend, xbin, ybin) = self.controller.get_roi()
        height = xend - x0
        width = yend - y0
        self.settings.child('roi', 'binning').setValue(xbin)
        self.settings.child('roi', 'width').setValue(width)
        self.settings.child('roi', 'height').setValue(height)

        self._prepare_view()

        info = "Initialized camera"
        initialized = True
        return info, initialized

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "camera_list":
            if self.controller != None:
                self.close()
            self.ini_detector(controller=self.controller)
        elif param.name() == "camera_user_id":
            try:
                self.controller.camera.DeviceUserID.SetValue(param.value())
                self.user_id = param.value()
                camera_list = [cam.GetUserDefinedName() for cam in BaslerCamera.list_cameras()]
                self.settings.child('ID', 'camera_list').setLimits(camera_list)
            except Exception:
                pass
        elif param.name() == "update_roi":
            if param.value():  # Switching on ROI

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
        elif param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child('roi', 'binning').value()
            ybin = self.settings.child('roi', 'binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)
        elif param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                wdet, hdet = self.controller.get_detector_size()
                # self.settings.child('ROIselect', 'x0').setValue(0)
                # self.settings.child('ROIselect', 'width').setValue(wdet)
                self.settings.child('roi', 'binning').setValue(1)
                #
                # self.settings.child('ROIselect', 'y0').setValue(0)
                # new_height = self.settings.child('ROIselect', 'height').setValue(hdet)

                new_roi = (0, wdet, 1, 0, hdet, 1)
                self.update_rois(new_roi)
                param.setValue(False)
        elif param.name() == "exposure_auto":
            try:
                par = getattr(self.controller.camera, self.controller.exposure_auto)
                par.SetIntValue(param.value())
            except Exception:
                pass
        elif param.name() == "exposure_time":
            try:
                par = getattr(self.controller.camera, self.controller.exposure_time)
                par.SetValue(param.value())
            except Exception:
                pass
        elif param.name() == "gain_auto":
            try:
                par = getattr(self.controller.camera, self.controller.gain_auto)
                par.SetIntValue(param.value())
            except Exception:
                pass
        elif param.name() == "gain_value":
            try:
                par = getattr(self.controller.camera, self.controller.gain_value)
                par.SetValue(param.value())
            except Exception:
                pass
        elif param.name() == "frame_rate":
            try:
                par = getattr(self.controller.camera, self.controller.frame_rate)
                par.SetValue(param.value())
            except Exception:
                pass
        elif param.name() == "gamma_enable":
            try:
                par = getattr(self.controller.camera, self.controller.gamma_enable)
                par.SetIntValue(param.value())
            except Exception:
                pass
        elif param.name() == "gamma_value":
            try:
                par = getattr(self.controller.camera, self.controller.gamma_value)
                par.SetIntValue(param.value())
            except Exception:
                pass
        elif param.name() == "gevscpd":
            try:
                par = getattr(self.controller.camera, self.controller.gevscpd)
                par.SetIntValue(param.value())
            except Exception:
                pass

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
 
         mock_data = np.zeros((width, height))

         self.x_axis = Axis(label='Pixels', data=np.linspace(1, width, width), index=0)
 
         if width != 1 and height != 1:
             data_shape = 'Data2D'
             self.y_axis = Axis(label='Pixels', data=np.linspace(1, height, height), index=1)
             self.axes = [self.x_axis, self.y_axis]
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
        #if not self.controller.camera.IsGrabbing():
        #    self.controller.camera.StartGrabbing()
        try:
            if live:
                self._prepare_view()
                self.controller.start_grabbing(frame_rate=self.settings.param('frame_rate').value())
            else:
                self._prepare_view()
                self.emit_data()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data(self):
        """
            Function used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            # Get data from buffer
            frame = self.controller.read_newest_image()
            # Emit the frame.
            if frame is not None:
                self.dte_signal.emit(
                    DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                        name=f'{self.user_id}',
                        data=[np.squeeze(frame)],
                        dim=self.data_shape,
                        labels=[f'{self.user_id}_{self.data_shape}'],
                        axes=self.axes)]))

            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def emit_data_callback(self, frame) -> None:
        self.dte_signal.emit(
            DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                axes=self.axes)]))

    def stop(self):
        self.controller.camera.StopGrabbing()
        return ''
    
    def close(self):
        """Terminate the communication protocol"""
        if self.controller.camera.IsGrabbing():
            self.controller.camera.StopGrabbing()
        self.controller.close()

        if self.callback_thread is not None:
            self.callback_thread.quit()
            self.callback_thread.wait()
            self.callback_thread = None

        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""           
        print(f"{self.user_id} communication terminated successfully")   
    
    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        sleep_ms = 100
        ind = 0

        while not self.controller.imageEventHandler.frame_ready:
            QtCore.QThread.msleep(sleep_ms)
            QtWidgets.QApplication.processEvents()

            ind += 1
            if ind * sleep_ms >= 1000:
                self.emit_status(ThreadCommand('_raise_timeout'))
                break

    # TODO Implement periodic temperature check
    def check_temp(self):
        try:
            temp = self.controller.camera.TemperatureAbs.Value
            self.settings.child('temp').setValue(temp)
        except Exception:
            pass


class BaslerCallback(QtCore.QObject):
    data_ready = pyqtSignal()  # Signal when data is ready

    def __init__(self, wait_fn):
        super().__init__()
        # Set the wait function
        self.wait_fn = wait_fn

    def wait_for_acquisition(self):
        new_data = self.wait_fn()
        if new_data is not False:  # will be returned if the main thread called CancelWait
            self.data_sig.emit()

if __name__ == '__main__':
    main(__file__, init=False)