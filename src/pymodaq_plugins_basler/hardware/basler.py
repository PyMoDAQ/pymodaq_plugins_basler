import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from numpy.typing import NDArray
from pypylon import pylon
from qtpy import QtCore
import json
import os

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


pixel_lengths: dict[str, float] = {
    # camera model name: pixel length in µm
    "daA1280-54um": 3.75,
    "daA2500-14um": 2.2,
    "daA3840-45um": 2,
    "acA640-120gm": 5.6,
    "acA645-100gm": 5.6,
    "acA1920-40gm": 5.86,
}


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
        self.model_name = None
        self.device_info = None


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

        self._pixel_length: Optional[float] = None
        self.attributes = {}
        self.open()
        if callback is not None:
            self.set_callback(callback=callback)

    def open(self) -> None:
        device = self.tlFactory.CreateDevice(self.model_name)
        self.camera.Attach(device)
        self.camera.Open()
        self.get_attributes()

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

    def get_device_info(self) -> List[Any]:
        """Get camera information.

        Return tuple ``(name, model, serial, devclass, devversion, vendor, friendly_name, user_name,
        props)``.
        """
        devInfo: pylon.DeviceInfo = self.camera.GetDeviceInfo()
        return [
            devInfo.GetFullName(),
            devInfo.GetModelName(),
            devInfo.GetSerialNumber(),
            devInfo.GetDeviceClass(),
            devInfo.GetDeviceVersion(),
            devInfo.GetVendorName(),
            devInfo.GetFriendlyName(),
            devInfo.GetUserDefinedName(),
            None,
        ]
    
    def get_attributes(self):
        devInfo = self.camera.GetDeviceInfo()
        self.model_name = devInfo.GetModelName()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f'config_{self.model_name}.json')
        with open(file_path, 'r') as file:
            self.attributes = json.load(file)
            self.gain_value = self.attributes["Gain"]["name"]
            self.exposure_time = self.attributes["Exposure Time"]["name"]
            self.exposure_auto = self.attributes["Exposure Auto"]["name"]
            self.gain_auto = self.attributes["Gain Auto"]["name"]
            self.gamma_enable = self.attributes["Gamma Enable"]["name"]
            self.gamma_value = self.attributes["Gamma"]["name"]
            self.frame_rate = self.attributes["Acquisition Frame Rate"]["name"]
            self.gevscpd = self.attributes["GevSCPD"]["name"]


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

    def clear_acquisition(self):
        """Stop acquisition"""
        pass  # TODO

    def setup_acquisition(self):
        self.camera.TriggerSelector.SetValue("AcquisitionStart")
        self.camera.TriggerMode.SetValue("Off")
        self.camera.TriggerSelector.SetValue("FrameStart")
        self.camera.TriggerMode.SetValue("Off")
        self.camera.AcquisitionFrameRateEnable.SetValue(False)
        self.camera.AcquisitionMode.SetValue("Continuous")

    def acquisition_in_progress(self):
        raise NotImplementedError("Not implemented")

    def read_newest_image(self):
        return self.get_one()

    def close(self) -> None:
        self.camera.Close()
        self.camera.DetachDevice()
        self._pixel_length = None

    # additional methods, for use in the code
    def get_single_result(self, timeout_ms: int = 1000) -> pylon.GrabResult:
        """Get one image and return the pylon `GrabResult`."""
        args = []
        if timeout_ms is not None:
            args.append(timeout_ms)
        if args:
            result: pylon.GrabResult = self.camera.GrabOne(*args)
        else:
            result = self.camera.GrabOne()
        if result.GrabSucceeded():
            return result
        else:
            raise TimeoutError("Grabbing exceeded timeout")

    def get_one(self, timeout_ms: int = 1000):
        """Get one image and return the (numpy) array of it."""
        result = self.get_single_result(timeout_ms=1000)
        result.GetArray()

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
            self.signals.imageGrabbed.emit(grabResult.GetArray())
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
                temp = self.controller.camera.TemperatureAbs.Value
                self.temperature_updated.emit(temp)
            except Exception:
                pass
            QtCore.QThread.msleep(self.interval)
        self.finished.emit()

def detector_clamp(value: Union[float, int], max_value: int) -> int:
    """Clamp a value to possible detector position."""
    return max(0, min(int(value), max_value))