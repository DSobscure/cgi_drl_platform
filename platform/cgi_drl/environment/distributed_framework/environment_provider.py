import msgpack
import numpy as np
from cgi_drl.environment.distributed_framework.socket_client import SocketClient
from cgi_drl.environment.distributed_framework.protocol import *
import pickle
import importlib

class EnvironmentProvider(SocketClient):
    def __init__(self, ip, port, requester_guid, environment_index):
        super().__init__(ip, port)
        print("############################################################")
        print("# Remote Environment Provider")
        print("# Connect to port: {}".format(self.port))
        print("############################################################")
        self._report_identity(requester_guid, environment_index)

        while True:
            self._wait_command()

    def _send_environment_provider_request(self, operation_code, parameters):
        self._send_operation_request(OperationCode.EnvironmentProviderRequest, 
            {
                SubRequestParameterCode.SubRequestCode : operation_code,
                SubRequestParameterCode.SubRequestParameters : msgpack.packb(parameters, use_bin_type=True)
            }
        )

    def _receive_environment_provider_response(self):
        operationCode, returnCode, parameters, operationMessage = self._receive_operation_response()
        if operationCode == OperationCode.EnvironmentProviderRequest and returnCode == OperationReturnCode.Successiful:
            sub_request_code = parameters[SubRequestResponseParameterCode.SubRequestCode]
            sub_request_return_code = OperationReturnCode(parameters[SubRequestResponseParameterCode.SubRequestReturnCode])
            sub_request_response_parameters = parameters[SubRequestResponseParameterCode.SubRequestResponseParameters]
            sub_request_operation_message = str(parameters[SubRequestResponseParameterCode.SubRequestOperationMessage])
            return sub_request_code, sub_request_return_code, sub_request_response_parameters, sub_request_operation_message
        else:
            print("EnvironmentProvider response error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def _receive_environment_provider_request(self):
        operationCode, parameters = self._receive_operation_request()
        if operationCode == OperationCode.EnvironmentProviderRequest:
            sub_request_code = parameters[SubRequestParameterCode.SubRequestCode]
            sub_request_parameters = parameters[SubRequestParameterCode.SubRequestParameters]
            return sub_request_code, sub_request_parameters
        else:
            print("EnvironmentProvider request error operationCode: {}".format(operationCode))
            exit(0)

    def _send_environment_provider_response(self, operation_code, return_code, parameters, operation_message):
        self._send_operation_response(
            OperationCode.EnvironmentProviderRequest, 
            OperationReturnCode.Successiful, 
            {
                SubRequestResponseParameterCode.SubRequestCode : operation_code,
                SubRequestResponseParameterCode.SubRequestReturnCode : return_code,
                SubRequestResponseParameterCode.SubRequestResponseParameters : msgpack.packb(parameters, use_bin_type=True),
                SubRequestResponseParameterCode.SubRequestOperationMessage : operation_message
            },
            "")

    def _report_identity(self, requester_guid, environment_index):
        self._send_operation_request(OperationCode.ReportIdentity, {
            ReportIdentityRequestParameterCode.Identity : IdentityCode.EnvironmentProvider,
            ReportIdentityRequestParameterCode.RequesterGuid : requester_guid,
            ReportIdentityRequestParameterCode.EnvironmentIndex : environment_index
        })
        # operationCode, returnCode, parameters, operationMessage = self._receive_operation_response()
        # if operationCode != OperationCode.ReportIdentity or returnCode != OperationReturnCode.Successiful:
        #     print("ReportIdentity error, OperationCode: {}, ReturnCode: {}, OperationMessage: {}".format(operationCode, returnCode, operationMessage))
        #     exit(0)

    def _wait_command(self):
        request_code, request_parameters = self._receive_environment_provider_request()

        if request_code == EnvironmentProviderOperationCode.Launch:
            environment_index = int(request_parameters[LaunchProviderRequestParameterCode.EnvironmentIndex])
            config = pickle.loads(request_parameters[LaunchProviderRequestParameterCode.Config])
            config["environment_index"] = environment_index
            agent_count = self.launch(config)
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.Launch, 
                OperationReturnCode.Successiful, 
                {LaunchProviderResponseParameterCode.AgentCount : agent_count}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.GetActionSpace:
            action_space = self.get_action_space()
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.GetActionSpace, 
                OperationReturnCode.Successiful, 
                {GetActionSpaceProviderResponseParameterCode.ActionSpace : pickle.dumps(action_space)}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.Reset:
            reset_settings = pickle.loads(request_parameters[ResetProviderRequestParameterCode.ResetSettings])
            observation = self.reset(reset_settings)
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.Reset, 
                OperationReturnCode.Successiful, 
                {ResetProviderResponseParameterCode.Observation : pickle.dumps(observation)}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.GetActionSpace:
            action_space = self.get_action_space()
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.GetActionSpace, 
                OperationReturnCode.Successiful, 
                {GetActionSpaceProviderResponseParameterCode.ActionSpace : pickle.dumps(action_space)}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.GetTurn:
            turn = self.get_turn()
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.GetTurn, 
                OperationReturnCode.Successiful, 
                {GetTurnProviderResponseParameterCode.Turn : pickle.dumps(turn)}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.SampleAction:
            action = self.sample()
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.SampleAction, 
                OperationReturnCode.Successiful, 
                {SampleActionProviderResponseParameterCode.Action : pickle.dumps(action)}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.Step:
            action_parameters = pickle.loads(request_parameters[StepProviderRequestParameterCode.ActionParameters])
            actions = action_parameters["actions"]
            action_settings = action_parameters["action_settings"]
            observations, rewards, dones, infos = self.step(actions, action_settings)
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.Step, 
                OperationReturnCode.Successiful, 
                {
                    StepProviderResponseParameterCode.Observation : pickle.dumps(observations),
                    StepProviderResponseParameterCode.Reward : pickle.dumps(rewards),
                    StepProviderResponseParameterCode.Done : pickle.dumps(dones),
                    StepProviderResponseParameterCode.Info : pickle.dumps(infos)
                }, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.ServerLaunch:
            environment_index = int(request_parameters[ServerLaunchProviderRequestParameterCode.EnvironmentIndex])
            config = pickle.loads(request_parameters[ServerLaunchProviderRequestParameterCode.Config])
            config["environment_index"] = environment_index
            agent_count = self.launch(config)
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.ServerLaunch, 
                OperationReturnCode.Successiful, 
                {ServerLaunchProviderResponseParameterCode.AgentCount : agent_count}, 
                ""
            )
        elif request_code == EnvironmentProviderOperationCode.NeedRestart:
            result = self.need_restart()
            self._send_environment_provider_response(
                EnvironmentProviderOperationCode.NeedRestart, 
                OperationReturnCode.Successiful, 
                {NeedRestartProviderResponseParameterCode.Result : result}, 
                ""
            )
        else:
            print("Error,Provider receive {}".format(request_code))
            exit(0)

    def launch(self, config):
        EnvironmentWrapper = getattr(importlib.import_module(config["environment_wrapper_path"]), config["environment_wrapper_class_name"])
        self.environment_wrapper = EnvironmentWrapper(config)
        return self.environment_wrapper.get_agent_count()

    def get_action_space(self):
        return self.environment_wrapper.get_action_space()

    def reset(self, reset_settings):
        observations = self.environment_wrapper.reset(reset_settings=reset_settings)
        return observations

    def get_turn(self):
        return self.environment_wrapper.get_turn()

    def sample(self):
        return self.environment_wrapper.sample()

    def step(self, actions, action_settings):
        observations, rewards, dones, infos = self.environment_wrapper.step(actions, action_settings)
        return observations, rewards, dones, infos

    def close(self):
        if self.environment_wrapper != None:
            self.environment_wrapper.close()
            self.environment_wrapper = None
        super().close()

    def need_restart(self):
        return self.environment_wrapper.need_restart()

if __name__ == '__main__':
    import sys
    provider = EnvironmentProvider(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])