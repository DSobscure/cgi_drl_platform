from enum import IntEnum

# base protocol
class IdentityCode(IntEnum):
    EnvironmentProvider = 0
    EnvironmentRequester = 1

class OperationReturnCode(IntEnum):
    UndefinedError = 0
    Successiful = 1
    ParameterCountError = 2
    NullObject = 3
    DbTransactionFailed = 4
    DbNoChanged = 5
    ParameterFormateError = 6
    Duplicated = 7
    NotExisted = 8
    AuthenticationFailed = 9
    InstantiateFailed = 10,
    OutOfResource = 11

class SubRequestParameterCode(IntEnum):
    SubRequestCode = 0
    SubRequestParameters = 1

class SubRequestResponseParameterCode(IntEnum):
    SubRequestCode = 0
    SubRequestReturnCode = 1
    SubRequestResponseParameters = 2
    SubRequestOperationMessage = 3

# socket level protocol
class OperationCode(IntEnum):
    ReportIdentity = 0
    EnvironmentProviderRequest = 1
    EnvironmentRequesterRequest = 2

class ReportIdentityRequestParameterCode(IntEnum):
    Identity = 0
    RequesterGuid = 1
    EnvironmentIndex = 2

# environment level protocol
class EnvironmentRequesterOperationCode(IntEnum):
    AllocateEnvironment = 0
    Launch = 1
    GetActionSpace = 2
    ResetAll = 3
    Reset = 4
    GetTurn = 5
    SampleActionAll = 6
    SampleAction = 7
    Step = 8
    RestartEnvironment = 9
    NeedRestart = 10

class EnvironmentProviderOperationCode(IntEnum):
    Launch = 0
    GetActionSpace = 1
    Reset = 2
    GetTurn = 3
    SampleAction = 4
    Step = 5
    ServerLaunch = 6
    NeedRestart = 7
    
# AllocateEnvironment
class AllocateEnvironmentRequesterRequestParameterCode(IntEnum):
    EnvironmentCount = 0
class AllocateEnvironmentProviderRequestParameterCode(IntEnum):
    RequesterGuid = 0

# Launch
class LaunchRequesterRequestParameterCode(IntEnum):
    Config = 0
class LaunchProviderRequestParameterCode(IntEnum):
    EnvironmentIndex = 0
    Config = 1
class LaunchRequesterResponseParameterCode(IntEnum):
    AgentCounts = 0
class LaunchProviderResponseParameterCode(IntEnum):
    AgentCount = 0

# GetActionSpace
class GetActionSpaceRequesterRequestParameterCode(IntEnum):
    Index = 0
class GetActionSpaceRequesterResponseParameterCode(IntEnum):
    ActionSpace = 0
class GetActionSpaceProviderResponseParameterCode(IntEnum):
    ActionSpace = 0

# Reset
class ResetAllRequesterRequestParameterCode(IntEnum):
    ResetSettings = 0
class ResetRequesterRequestParameterCode(IntEnum):
    Index = 0
    ResetSettings = 1
class ResetProviderRequestParameterCode(IntEnum):
    ResetSettings = 0
class ResetAllRequesterResponseParameterCode(IntEnum):
    Observations = 0
class ResetRequesterResponseParameterCode(IntEnum):
    Observation = 0
class ResetProviderResponseParameterCode(IntEnum):
    Observation = 0

# GetTurn
class GetTurnRequesterResponseParameterCode(IntEnum):
    Turns = 0
class GetTurnProviderResponseParameterCode(IntEnum):
    Turn = 0

# SampleAction
class SampleActionRequesterRequestParameterCode(IntEnum):
    Index = 0
class SampleActionAllRequesterResponseParameterCode(IntEnum):
    Actions = 0
class SampleActionRequesterResponseParameterCode(IntEnum):
    Action = 0
class SampleActionProviderResponseParameterCode(IntEnum):
    Action = 0

# Step
class StepRequesterRequestParameterCode(IntEnum):
    ActionParameters = 0
class StepProviderRequestParameterCode(IntEnum):
    ActionParameters = 0
class StepRequesterResponseParameterCode(IntEnum):
    Observations = 0
    Rewards = 1
    Dones = 2
    Infos = 3
class StepProviderResponseParameterCode(IntEnum):
    Observation = 0
    Reward = 1
    Done = 2
    Info = 3

# RestartEnvironment
class RestartEnvironmentRequesterRequestParameterCode(IntEnum):
    Index = 0
    Config = 1

# RestartEnvironment
class ServerLaunchProviderRequestParameterCode(IntEnum):
    EnvironmentIndex = 0
    Config = 1
class ServerLaunchProviderResponseParameterCode(IntEnum):
    AgentCount = 0

# NeedRestart
class NeedRestartRequesterRequestParameterCode(IntEnum):
    Index = 0
class NeedRestartRequesterResponseParameterCode(IntEnum):
    Result = 0
class NeedRestartProviderResponseParameterCode(IntEnum):
    Result = 0
