from __future__ import annotations
from typing import Optional, Union

from easydict import EasyDict as edict

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

from pymodaq_utils.utils import ThreadCommand, getLineInfo

from pymodaq_gui.parameter import Parameter
from pymodaq_utils.serialize.serializer_legacy import DeSerializer
from pymodaq_gui.parameter import ioxml
import pymodaq_gui.parameter.utils as putils

from pymodaq.utils.leco.leco_director import LECODirector, leco_parameters
from pymodaq.utils.leco.director_utils import DetectorDirector


class DAQ_2DViewer_BaslerLECO(LECODirector, DAQ_Viewer_base):
    """A control module, which in the dashboard, allows to control a remote Viewer module.

    This is the base class for the viewer LECO director modules.
    """

    settings: Parameter
    controller: DetectorDirector

    params_GRABBER = [{'title': 'Actor Settings', 'name': 'settings_client', 'type': 'group', 'children': [
                    {'title': 'Exposure Settings', 'name': 'exposure', 'type': 'group', 'children': [
                        {"title": "Exposure Time (ms)", "name": "ExposureTimeRaw", "type": "int", "value": 100.0, "default": 100.0, "limits": [0.0, 10000.0]}
                    ]},
                    {"title": "Gain Settings", "name": "gain", "type": "group", "children": [
                        {"title": "Gain Value", "name": "GainRaw", "type": "int", "value": 1, "default": 1, "limits": [0, 240]}
                    ]},
                    {"title": "Trigger Settings", "name": "trigger", "type": "group", "children": [
                        {"title": "Trigger Mode", "name": "TriggerMode", "type": "led_push", "value": False}
                    ]},
                    {"title": "Additional Settings", "name": "misc", "type": "group", "children": [
                        {"title": "Gev SCPS Packet Size", "name": "GevSCPSPacketSize", "type": "int", "value": 3000, "limits": [0, 6000]}
                    ]}]}]

    message_list = LECODirector.message_list + ["Quit", "Send Data 0D", "Send Data 1D",
                                                "Send Data 2D", "Send Data ND",
                                                "Status", "Done", "Server Closed",
                                                "Info", "Infos", "Info_xml", 'x_axis', 'y_axis']
    socket_types = ["GRABBER"]
    params = comon_parameters + leco_parameters + params_GRABBER

    def __init__(self, parent=None, params_state=None, grabber_type: str = "0D", **kwargs) -> None:
        super().__init__(parent=parent, params_state=params_state, **kwargs)
        self.register_rpc_methods((
            self.set_x_axis,
            self.set_y_axis,
        ))
        for method in (
            self.set_data,
        ):
            self.listener.register_binary_rpc_method(method, accept_binary_input=True)

        self.client_type = "GRABBER"
        self.x_axis = None
        self.y_axis = None
        self.grabber_type = grabber_type

    def ini_detector(self, controller=None):
        """
            | Initialisation procedure of the detector updating the status dictionary.
            |
            | Init axes from image , here returns only None values (to tricky to di it with the
              server and not really necessary for images anyway)

            See Also
            --------
            utility_classes.DAQ_TCP_server.init_server, get_xaxis, get_yaxis
        """
        self.status.update(edict(initialized=False, info="", x_axis=None, y_axis=None,
                                 controller=None))
        actor_name = self.settings.child("actor_name").value()
        self.controller = self.ini_detector_init(  # type: ignore
            old_controller=controller,
            new_controller=DetectorDirector(actor=actor_name, communicator=self.communicator),
            )
        self.controller.set_remote_name(self.communicator.full_name)  # type: ignore
        try:
            # self.settings.child(('infos')).addChildren(self.params_GRABBER)

            # init axes from image , here returns only None values (to tricky to di it with the
            # server and not really necessary for images anyway)
            self.x_axis = self.get_xaxis()
            self.y_axis = self.get_yaxis()
            self.status.x_axis = self.x_axis
            self.status.y_axis = self.y_axis
            self.status.initialized = True
            return self.status

        except Exception as e:
            self.status.info = getLineInfo() + str(e)
            self.status.initialized = False
            return self.status
        
    def commit_settings(self, param) -> None:
        self.commit_leco_settings(param=param)

    def get_xaxis(self):
        pass
        return self.x_axis

    def get_yaxis(self):
        pass
        return self.y_axis

    def grab_data(self, Naverage = 1, metadata: dict = None, save: bool = False, **kwargs):
        try:
            self.controller.set_remote_name(self.communicator.full_name)
            # Set the metadata of the underlying actor
            if save:
                # Make sure save_frame_local is False and save_frame_leco is True
                self.controller.set_parameters(parameters={"save_frame_local": False})
                self.controller.set_parameters(parameters={"save_frame_leco": True})
            if metadata is not None:
                print(metadata)
                self.controller.set_parameters(parameters={"metadata": metadata})                 
            self.controller.send_data(grabber_type=self.grabber_type)
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [getLineInfo() + str(e), "log"]))

    def stop(self):
        """
            not implemented.
        """
        pass
        return ""

    # Methods for RPC calls
    def set_x_axis(self, data, label: str = "", units: str = ""):
        # TODO make to work
        self.x_axis = dict(data=data, label=label, units=units)
        self.emit_x_axis()

    def set_y_axis(self, data, label: str = "", units: str = ""):
        # TODO make to work
        self.y_axis = dict(data=data, label=label, units=units)
        self.emit_y_axis()

    def set_data(self, data: Union[list, str, None],
                 additional_payload: Optional[list[bytes]]=None) -> None:
        """
        Set the grabbed data signal.

        corresponds to the "data_ready" signal

        :param data: If None, look for the additional object
        """
        if isinstance(data, str):
            deserializer = DeSerializer.from_b64_string(data)
        elif additional_payload is not None:
            deserializer = DeSerializer(additional_payload[0])
        else:
            raise NotImplementedError("Not implemented to set a list of values.")
        dte = deserializer.dte_deserialization()
        self.dte_signal.emit(dte)


if __name__ == '__main__':
    main(__file__, init=False)