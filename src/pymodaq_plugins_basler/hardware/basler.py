import logging
from typing import Any, Callable, List, Optional, Tuple, Union
import platform

from numpy.typing import NDArray
from pypylon import pylon
from qtpy import QtCore
import json
import os

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class BaslerCamera:
    """Control a Basler Dart camera in the style of pylablib.

    It wraps an :class:`pylon.InstantCamera` instance.

    :param name: Full name of the device.
    :param callback: Callback method for each grabbed image
    """

    tlFactory: pylon.TlFactory
    camera: pylon.InstantCamera

    def __init__(self, info: str, callback: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        # create camera object
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.camera = pylon.InstantCamera()
        self.model_name = info.GetModelName()
        self.device_info = info

        # Default place to look for saved device configuration
        if platform.system() == 'Windows':
            self.default_device_state_path = os.path.join(
                os.environ.get('PROGRAMDATA'), '.pymodaq', f'{self.model_name}_config.pfs'
            )
        else:
            self.default_device_state_path = os.path.join(
                '/etc', '.pymodaq', f'{self.model_name}_config.pfs'
            )

        # register configuration event handler
        self.configurationEventHandler = ConfigurationHandler()
        self.camera.RegisterConfiguration(
            self.configurationEventHandler,
            pylon.RegistrationMode_ReplaceAll,
            pylon.Cleanup_None,
        )
        # configure camera events
        self.imageEventHandler = ImageEventHandler()
        self.camera.RegisterImageEventHandler(
            self.imageEventHandler, pylon.RegistrationMode_Append, pylon.Cleanup_None
        )

        self.imageEventHandler.signals.imageGrabbed.connect(lambda x: print("Image grabbed"))

        self.attributes = {}
        self.open()
        if callback is not None:
            self.set_callback(callback=callback)

    def open(self) -> None:
        device = self.tlFactory.CreateDevice(self.device_info.GetFullName())
        self.camera.Attach(device)
        self.camera.Open()
        self.get_attributes()
        self.attribute_names = [attr['name'] for attr in self.attributes] + [child['name'] for attr in self.attributes if attr.get('type') == 'group' for child in attr.get('children', [])]

    def set_callback(
        self, callback: Callable[[NDArray], None], replace_all: bool = True
    ) -> None:
        """Setup a callback method for continuous acquisition.

        :param callback: Method to be used in continuous mode. It should accept an array as input.
        :param bool replace_all: Whether to remove all previously set callback methods.
        """
        if replace_all:
            try:
                self.imageEventHandler.signals.imageGrabbed.disconnect()
            except TypeError:
                pass  # not connected
        self.imageEventHandler.signals.imageGrabbed.connect(callback)

    # Methods in the style of pylablib
    @staticmethod
    def list_cameras() -> List[pylon.InstantCamera]:
        """List all available cameras as camera info objects."""
        tlFactory = pylon.TlFactory.GetInstance()
        return tlFactory.EnumerateDevices()
    

    def get_attributes(self):
        """Get the attributes of the camera and store them in a dictionary."""
        name = self.model_name.replace(" ", "-")

        if platform.system() == 'Windows':
            base_dir = os.path.join(os.environ.get('PROGRAMDATA'), '.pymodaq')
        else:
            base_dir = '/etc/.pymodaq'

        file_path = os.path.join(base_dir, f'config_{name}.json')

        with open(file_path, 'r') as file:
            attributes = json.load(file)
            self.attributes = self.clean_device_attributes(attributes)


    def get_roi(self) -> Tuple[float, float, float, float, int, int]:
        """Return x0, width, y0, height, xbin, ybin."""
        x0 = self.camera.OffsetX.GetValue()
        width = self.camera.Width.GetValue()
        y0 = self.camera.OffsetY.GetValue()
        height = self.camera.Height.GetValue()
        xbin = self.camera.BinningHorizontal.GetValue()
        ybin = self.camera.BinningVertical.GetValue()
        return x0, x0 + width, y0, y0 + height, xbin, ybin

    def set_roi(
        self, hstart: int, hend: int, vstart: int, vend: int, hbin: int, vbin: int
    ) -> None:
        camera = self.camera
        m_width, m_height = self.get_detector_size()
        inc = camera.Width.Inc  # minimum step size
        hstart = detector_clamp(hstart, m_width) // inc * inc
        vstart = detector_clamp(vstart, m_height) // inc * inc
        # Set the offset to 0 first, to allow full range of width values.
        camera.OffsetX.SetValue(0)
        camera.Width.SetValue((detector_clamp(hend, m_width) - hstart) // inc * inc)
        camera.OffsetX.SetValue(hstart)
        camera.OffsetY.SetValue(0)
        camera.Height.SetValue((detector_clamp(vend, m_height) - vstart) // inc * inc)
        camera.OffsetY.SetValue(vstart)
        camera.BinningHorizontal.SetValue(int(hbin))
        camera.BinningVertical.SetValue(int(vbin))

    def get_detector_size(self) -> Tuple[int, int]:
        """Return width and height of detector in pixels."""
        return self.camera.SensorWidth.GetValue(), self.camera.SensorHeight.GetValue()

    def get_attribute_value(self, name, error_on_missing=True):
        """Get the camera attribute with the given name"""
        return self.attributes[name]

    def setup_acquisition(self):
        self.camera.TriggerSelector.SetValue("AcquisitionStart")
        self.camera.TriggerMode.SetValue("Off")
        self.camera.TriggerSelector.SetValue("FrameStart")
        self.camera.TriggerMode.SetValue("Off")
        self.camera.AcquisitionFrameRateEnable.SetValue(False)
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.AcquisitionFrameRateEnable.SetValue(True)

    def close(self) -> None:
        self.camera.Close()
        self.camera.DetachDevice()

    def start_grabbing(self, frame_rate: int) -> None:
        """Start continuously to grab data.

        Whenever a grab succeeded, the callback defined in :meth:`set_callback` is called.
        """
        try:
            self.camera.AcquisitionFrameRate.SetValue(frame_rate)
        except pylon.LogicalErrorException:
            pass
        self.camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByInstantCamera
        )

    def stop_grabbing(self):
        self.camera.StopGrabbing()
        return ''

    def save_device_state(self):
        save_path = self.default_device_state_path
        node_map = self.camera.GetNodeMap()
        try:
            pylon.FeaturePersistence.Save(save_path, node_map)
            print(f"Device state saved to {save_path}")
        except Exception as e:
            print(f"Failed to save device state: {e}")

    def load_device_state(self):
        load_path = self.default_device_state_path
        node_map = self.camera.GetNodeMap()
        if os.path.isfile(load_path):
            try:
                pylon.FeaturePersistence.Load(load_path, node_map)
                print(f"Device state loaded")
            except Exception as e:
                print(f"Failed to load device state: {e}")
        else:
            print("No saved settings file found to load.")

    
    def clean_device_attributes(self, attributes):
        clean_params = []

        # Check if attributes is a list or dictionary
        if isinstance(attributes, dict):
            items = attributes.items()
        elif isinstance(attributes, list):
            # If it's a list, we assume each item is a parameter (no keys)
            items = enumerate(attributes)  # Use index for 'key'
        else:
            raise ValueError(f"Unsupported type for attributes: {type(attributes)}")

        for idx, attr in items:
            param = {}

            param['title'] = attr.get('title', '')
            param['name'] = attr.get('name', str(idx))  # use index if name is missing
            param['type'] = attr.get('type', 'str')
            param['value'] = attr.get('value', '')
            param['default'] = attr.get('default', None)
            param['limits'] = attr.get('limits', None)
            param['readonly'] = attr.get('readonly', False)

            if param['type'] == 'group' and 'children' in attr:
                children = attr['children']
                # If children is a dict, convert to a list
                if isinstance(children, dict):
                    children = list(children.values())
                param['children'] = self.clean_device_attributes(children)

            clean_params.append(param)

        return clean_params

class ConfigurationHandler(pylon.ConfigurationEventHandler):
    """Handle the configuration events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ConfigurationHandlerSignals()

    class ConfigurationHandlerSignals(QtCore.QObject):
        """Signals for the CameraEventHandler."""

        cameraRemoved = QtCore.pyqtSignal(object)

    def OnOpened(self, camera: pylon.InstantCamera) -> None:
        """Standard configuration after being opened."""
        camera.PixelFormat.SetValue("Mono12")
        camera.GainAuto.SetValue("Off")
        camera.ExposureAuto.SetValue("Off")

    def OnCameraDeviceRemoved(self, camera: pylon.InstantCamera) -> None:
        """Emit a signal that the camera is removed."""
        self.signals.cameraRemoved.emit(camera)


class ImageEventHandler(pylon.ImageEventHandler):
    """Handle the events and translates them to signals/slots."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ImageEventHandlerSignals()
        self.frame_ready = False

    class ImageEventHandlerSignals(QtCore.QObject):
        """Signals for the ImageEventHandler."""

        imageGrabbed = QtCore.pyqtSignal(object)

    def OnImageSkipped(self, camera: pylon.InstantCamera, countOfSkippedImages: int) -> None:
        """Handle a skipped image."""
        log.warning(f"{countOfSkippedImages} images have been skipped.")

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grabResult: pylon.GrabResult) -> None:
        """Process a grabbed image."""
        if grabResult.GrabSucceeded():
            self.frame_ready = True
            frame_data = {"frame": grabResult.GetArray(), "timestamp": grabResult.GetTimeStamp()}
            self.signals.imageGrabbed.emit(frame_data)
        else:
            log.warning(
                (
                    f"Grab failed with code {grabResult.GetErrorCode()}, "
                    f"{grabResult.GetErrorDescription()}."
                )
            )


class TemperatureMonitor(QtCore.QObject):
    temperature_updated = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self, camera_handle, check_interval=100):
        super().__init__()
        self._running = True
        self.camera = camera_handle
        self.interval = check_interval

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                temp = self.camera.TemperatureAbs.Value
                self.temperature_updated.emit(temp)
            except Exception:
                pass
            QtCore.QThread.msleep(self.interval)
        self.finished.emit()

def detector_clamp(value: Union[float, int], max_value: int) -> int:
    """Clamp a value to possible detector position."""
    return max(0, min(int(value), max_value))