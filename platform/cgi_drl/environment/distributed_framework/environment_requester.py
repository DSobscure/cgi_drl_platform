import msgpack
import numpy as np
from cgi_drl.environment.distributed_framework.socket_client import SocketClient
from cgi_drl.environment.distributed_framework.protocol import *
import pickle

class EnvironmentRequester(SocketClient):
    def __init__(self, config):
        self.config = config

        self.environment_host_ip = config["environment_host_ip"]
        self.environment_host_port = config["environment_host_port"]
        self.environment_count = config["environment_count"]

        
        super().__init__(self.environment_host_ip, self.environment_host_port)
        self._report_identity()
        self.allocate_environment(self.environment_count)

        self.agent_count = 0
        self.individual_agent_counts = []
        agent_counts = self.launch(config)
            
        for i in range(self.environment_count):
            self.individual_agent_counts.append(agent_counts[i])
            self.agent_count += agent_counts[i]

    def look_up_environment_and_agent_index(self, index):
        agent_index = index
        for env_index in range(self.environment_count):
            if agent_index < self.individual_agent_counts[env_index]:
                break
            else:
                agent_index -= self.individual_agent_counts[env_index]
        return env_index, agent_index

    def _send_environment_requester_request(self, operation_code, parameters):
        self._send_operation_request(OperationCode.EnvironmentRequesterRequest, 
            {
                SubRequestParameterCode.SubRequestCode : operation_code,
                SubRequestParameterCode.SubRequestParameters : msgpack.packb(parameters, use_bin_type=True)
            }
        )

    def _receive_environment_requester_response(self):
        operationCode, returnCode, parameters, operationMessage = self._receive_operation_response()
        if operationCode == OperationCode.EnvironmentRequesterRequest and returnCode == OperationReturnCode.Successiful:
            sub_request_code = parameters[SubRequestResponseParameterCode.SubRequestCode]
            sub_request_return_code = OperationReturnCode(parameters[SubRequestResponseParameterCode.SubRequestReturnCode])
            sub_request_response_parameters = parameters[SubRequestResponseParameterCode.SubRequestResponseParameters]
            sub_request_operation_message = str(parameters[SubRequestResponseParameterCode.SubRequestOperationMessage])
            return sub_request_code, sub_request_return_code, sub_request_response_parameters, sub_request_operation_message
        else:
            print("EnvironmentRequester response error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def _report_identity(self):
        self._send_operation_request(OperationCode.ReportIdentity, {
            ReportIdentityRequestParameterCode.Identity : IdentityCode.EnvironmentRequester,
            ReportIdentityRequestParameterCode.RequesterGuid : "0",
            ReportIdentityRequestParameterCode.EnvironmentIndex : 0
        })

    def allocate_environment(self, environment_count):
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.AllocateEnvironment, 
            {
                AllocateEnvironmentRequesterRequestParameterCode.EnvironmentCount : environment_count,
            }
        )

        operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
        if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.AllocateEnvironment and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
            self.environment_count = environment_count
        else:
            print("AllocateEnvironment error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def launch(self, config):
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.Launch, 
            {
                LaunchRequesterRequestParameterCode.Config : pickle.dumps(config),
            }
        )

        operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
        if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.Launch and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
            return parameters[LaunchRequesterResponseParameterCode.AgentCounts]
        else:
            print("Launch error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def get_action_space(self, index=None):
        if index == None:
            index = 0
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.GetActionSpace, 
            {
                GetActionSpaceRequesterRequestParameterCode.Index : index,
            }
        )

        operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
        if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.GetActionSpace and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
            return pickle.loads(parameters[GetActionSpaceRequesterResponseParameterCode.ActionSpace])
        elif EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.GetActionSpace and OperationReturnCode(returnCode) == OperationReturnCode.NotExisted:
            return None
        else:
            print("GetActionSpace error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)


    def reset(self, index=-1, reset_settings=None):
        if reset_settings == None:
            reset_settings = {}

        if index == -1:
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.ResetAll, 
                {
                    ResetAllRequesterRequestParameterCode.ResetSettings : pickle.dumps(reset_settings),
                }
            )

            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.ResetAll and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                observations = []
                for i, observation in enumerate(parameters[ResetAllRequesterResponseParameterCode.Observations]):
                    if observation == None:
                        observations.extend([{} for _ in range(self.individual_agent_counts[i])])
                    else:
                        observations.extend(pickle.loads(observation))
                return observations
            else:
                print("ReserAll error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)
        else:
            env_index, agent_index = self.look_up_environment_and_agent_index(index)
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.Reset, 
                {
                    ResetRequesterRequestParameterCode.Index : env_index,
                    ResetRequesterRequestParameterCode.ResetSettings : pickle.dumps(reset_settings),
                }
            )
            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.Reset and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                return pickle.loads(parameters[ResetRequesterResponseParameterCode.Observation])
            elif EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.Reset and OperationReturnCode(returnCode) == OperationReturnCode.NotExisted:
                return [{} for _ in range(self.individual_agent_counts[env_index])]
            else:
                print("Reset error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)

    def get_turn(self):
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.GetTurn, {})

        operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
        if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.GetTurn and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
            turns = []
            for i, turn in enumerate(parameters[GetTurnRequesterResponseParameterCode.Turns]):
                if turn == None:
                    turns.extend([False for _ in range(self.individual_agent_counts[i])])
                else:
                    turns.extend(pickle.loads(turn))
            return turns
        else:
            print("GetTurn error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def sample(self, index=-1):
        if index == -1:
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.SampleActionAll,{})

            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.SampleActionAll and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                actions = []
                for i, action in enumerate(parameters[SampleActionAllRequesterResponseParameterCode.Actions]):
                    if action == None:
                        actions.extend([None for _ in range(self.individual_agent_counts[i])])
                    else:
                        actions.extend([pickle.loads(action)])
                return actions
            else:
                print("SampleActionAll error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)
        else:
            env_index, agent_index = self.look_up_environment_and_agent_index(index)
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.SampleAction, 
                {
                    SampleActionRequesterRequestParameterCode.Index : env_index,
                }
            )
            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.SampleAction and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                return pickle.loads(parameters[SampleActionRequesterResponseParameterCode.Action])[agent_index]
            elif EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.SampleAction and OperationReturnCode(returnCode) == OperationReturnCode.NotExisted:
                return None
            else:
                print("SampleAction error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)

    def step(self, actions, action_settings=None):
        if action_settings == None:
            action_settings = {}
        agent_offset = 0
        action_parameters = []
        for i in range(self.environment_count):
            env_actions = actions[agent_offset:agent_offset+self.individual_agent_counts[i]]
            action_parameter = pickle.dumps({
                "actions": env_actions,
                "action_settings": action_settings
            })
            action_parameters.append(action_parameter)
            agent_offset += self.individual_agent_counts[i]
        
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.Step, 
            {
                StepRequesterRequestParameterCode.ActionParameters : action_parameters,
            }
        )

        operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
        if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.Step and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
            observations, rewards, dones, infos = [], [], [], []
            for i, observation in enumerate(parameters[StepRequesterResponseParameterCode.Observations]):
                if observation == None:
                    observations.extend([{} for _ in range(self.individual_agent_counts[i])])
                else:
                    observations.extend(pickle.loads(observation))
            for i, reward in enumerate(parameters[StepRequesterResponseParameterCode.Rewards]):
                if reward == None:
                    rewards.extend([0 for _ in range(self.individual_agent_counts[i])])
                else:
                    rewards.extend(pickle.loads(reward))
            for i, done in enumerate(parameters[StepRequesterResponseParameterCode.Dones]):
                if done == None:
                    dones.extend([False for _ in range(self.individual_agent_counts[i])])
                else:
                    dones.extend(pickle.loads(done))
            for i, info in enumerate(parameters[StepRequesterResponseParameterCode.Infos]):
                if info == None:
                    infos.extend([{} for _ in range(self.individual_agent_counts[i])])
                else:
                    infos.extend(pickle.loads(info))
            return observations, rewards, dones, infos
        else:
            print("Step error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
            exit(0)

    def restart(self, index):
        self._send_environment_requester_request(EnvironmentRequesterOperationCode.RestartEnvironment, 
            {
                RestartEnvironmentRequesterRequestParameterCode.Index : index,
                RestartEnvironmentRequesterRequestParameterCode.Config : pickle.dumps(self.config),
            }
        )

    def need_restart(self, environment_index=-1):
        if environment_index == -1:
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.NeedRestart, 
                {
                    NeedRestartRequesterRequestParameterCode.Index : environment_index,
                }
            )

            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.NeedRestart and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                results = []
                for i, result in enumerate(parameters[NeedRestartRequesterResponseParameterCode.Result]):
                    if result == None:
                        results.append(True)
                    else:
                        results.append(result)
                return results
            elif EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.NeedRestart and OperationReturnCode(returnCode) == OperationReturnCode.NotExisted:
                return [True] * self.environment_count
            else:
                print("NeedRestart error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)
        else:
            self._send_environment_requester_request(EnvironmentRequesterOperationCode.NeedRestart, 
                {
                    NeedRestartRequesterRequestParameterCode.Index : environment_index,
                }
            )

            operationCode, returnCode, parameters, operationMessage = self._receive_environment_requester_response()
            if EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.NeedRestart and OperationReturnCode(returnCode) == OperationReturnCode.Successiful:
                return bool(parameters[NeedRestartRequesterResponseParameterCode.Result][0])
            elif EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.NeedRestart and OperationReturnCode(returnCode) == OperationReturnCode.NotExisted:
                return True
            else:
                print(EnvironmentRequesterOperationCode(operationCode) == EnvironmentRequesterOperationCode.NeedRestart, OperationReturnCode(returnCode) == OperationReturnCode.NotExisted)
                print("NeedRestart error returnCode: {}, operationMessage: {}".format(returnCode, operationMessage))
                exit(0)

# test
if __name__ == '__main__':
    env = EnvironmentRequester({
        "environment_host_ip" : "127.0.0.1",
        "environment_host_port" : 30000,
        "environment_wrapper_path" : "cgi_drl.environment.atari.atari_environment_wrapper",
        "environment_wrapper_class_name" : "AtariEnvironmentWrapper",
        
        "environment_count" : 4,
        "environment_id" : "BreakoutNoFrameskip-v4",
        "episode_life" : False,
    })

    env.reset()
    print(env.get_action_space())
    for i in range(1000):
        actions = env.sample()
        obs, rewards, dones, infos = env.step(actions)
        print(actions, dones, infos)
        for i_agent in range(len(dones)):
            if dones[i_agent]:
                env.reset(i_agent, reset_settings={"mode":"hard"})