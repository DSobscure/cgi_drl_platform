using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Cgi.VideoGame.Distributed.Server
{
    public class EnvironmentProvider
    {
        protected LocalPeer terminal;
        protected Guid requesterGuid;
        protected Process process;
        public Guid TerminalGuid { get { return terminal.Guid; } }
        public int AgentCount { get; set; }
        public EnvironmentRequester EnvironmentRequester { get; protected set; }
        public int EnvironmentIndex { get; protected set; }

        private event Action<EnvironmentProvider> onClosed;
        public event Action<EnvironmentProvider> OnClosed { add { onClosed += value; } remove { onClosed -= value; } }

        private event Action onTerminalReady;
        public event Action OnTerminalReady { add { onTerminalReady += value; } remove { onTerminalReady -= value; } }

        public bool IsReady { get { return terminal != null; } }

        internal EnvironmentProvider(int environmentIndex, Guid requesterGuid, Process process, EnvironmentRequester requester)
        {
            EnvironmentIndex = environmentIndex;
            this.requesterGuid = requesterGuid;
            this.process = process;
            EnvironmentRequester = requester;
        }

        public void Close()
        {
            try
            {
                terminal?.Close();
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.GetType()}");
                Logger.Instance.Error($"{this} : {ex.Source}");
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
            terminal = null;
            try
            {
                process?.Kill();
                process?.Dispose();
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.GetType()}");
                Logger.Instance.Error($"{this} : {ex.Source}");
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
            process = null;
            onClosed?.Invoke(this);
            onClosed = null;
        }

        public void SendResponse(OperationResponseParameter operationResponse)
        {
            terminal?.SendResponse(new OperationResponseParameter
            {
                operationCode = (byte)OperationCode.EnvironmentProviderRequest,
                returnCode = (short)OperationReturnCode.Successiful,
                parameters = new Dictionary<byte, object> {
                    { (byte)SubRequestResponseParameterCode.SubRequestCode, operationResponse.operationCode },
                    { (byte)SubRequestResponseParameterCode.SubRequestReturnCode, operationResponse.returnCode },
                    { (byte)SubRequestResponseParameterCode.SubRequestResponseParameters, operationResponse.parameters },
                    { (byte)SubRequestResponseParameterCode.SubRequestOperationMessage, operationResponse.operationMessage },
                },
                operationMessage = ""
            });
        }

        public void SendRequest(OperationRequestParameter operationRequest)
        {
            terminal?.SendRequest(new OperationRequestParameter
            {
                operationCode = (byte)OperationCode.EnvironmentProviderRequest,
                parameters = new Dictionary<byte, object> {
                    { (byte)SubRequestParameterCode.SubRequestCode, operationRequest.operationCode },
                    { (byte)SubRequestParameterCode.SubRequestParameters, operationRequest.parameters }
                }
            });
        }

        public OperationReturnCode BindTerminal(LocalPeer terminal, out string errorMessage)
        {
            this.terminal = terminal;
            var returnCode = EnvironmentProviderRepository.Instance.Add(terminal.Guid, this, out errorMessage);
            onTerminalReady?.Invoke();
            onTerminalReady = null;
            return returnCode;
        }

        public void Launch(int environmentIndex, object config, out string errorMessage)
        {
            if(IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.Launch,
                    parameters = new Dictionary<byte, object>
                    {
                        { (byte)LaunchRequestParameterCode.EnvironmentIndex, environmentIndex },
                        { (byte)LaunchRequestParameterCode.Config, config }
                    }
                });
                errorMessage = "";
            }
            else
            {
                errorMessage = "server launch";
                OnTerminalReady += () => {
                    SendRequest(new OperationRequestParameter
                    {
                        operationCode = (byte)EnvironmentProviderOperationCode.ServerLaunch,
                        parameters = new Dictionary<byte, object>
                        {
                            { (byte)ServerLaunchRequestParameterCode.EnvironmentIndex, environmentIndex },
                            { (byte)ServerLaunchRequestParameterCode.Config, config }
                        }
                    });
                };
            }
        }

        public void LaunchResponse(OperationReturnCode returnCode, object agentCount, string operationMessage)
        {
            EnvironmentRequester.LaunchResponse(EnvironmentIndex, returnCode, agentCount, operationMessage);
        }

        public bool GetActionSpace(out string errorMessage)
        {
            errorMessage = "";
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.GetActionSpace,
                    parameters = new Dictionary<byte, object>()
                });
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
            
        }
        public void GetActionSpaceResponse(OperationReturnCode returnCode, object actionSpace, string operationMessage)
        {
            EnvironmentRequester.GetActionSpaceResponse(returnCode, actionSpace, operationMessage);
        }

        public bool Reset(object resetSettings, out string errorMessage)
        {
            if (IsReady)
            {
                terminal.StartFastPooling();
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.Reset,
                    parameters = new Dictionary<byte, object>
                    {
                        { (byte)ResetRequestParameterCode.ResetSettings, resetSettings },
                    }
                });
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
        }
        public void ResetResponse(OperationReturnCode returnCode, object observation, string operationMessage)
        {
            EnvironmentRequester.ResetResponse(returnCode, observation, operationMessage);
            EnvironmentRequester.ResetAllResponse(EnvironmentIndex, returnCode, observation, operationMessage);
        }

        public bool GetTurn(out string errorMessage)
        {
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.GetTurn,
                    parameters = new Dictionary<byte, object>()
                });
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
        }
        public void GetTurnResponse(OperationReturnCode returnCode, object turn, string operationMessage)
        {
            EnvironmentRequester.GetTurnResponse(EnvironmentIndex, returnCode, turn, operationMessage);
        }

        public bool SampleAction(out string errorMessage)
        {
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.SampleAction,
                    parameters = new Dictionary<byte, object>()
                });
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
        }
        public void SampleActionResponse(OperationReturnCode returnCode, object action, string operationMessage)
        {
            EnvironmentRequester.SampleActionResponse(returnCode, action, operationMessage);
            EnvironmentRequester.SampleActionAllResponse(EnvironmentIndex, returnCode, action, operationMessage);
        }

        public bool Step(object actionParameters, out string errorMessage)
        {
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.Step,
                    parameters = new Dictionary<byte, object>
                {
                    { (byte)StepRequestParameterCode.ActionParameters, actionParameters },
                }
                });
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
        }
        public void StepResponse(OperationReturnCode returnCode, object observation, object reward, object done, object info, string operationMessage)
        {
            EnvironmentRequester.StepResponse(EnvironmentIndex, returnCode, observation, reward, done, info, operationMessage);
        }

        public OperationReturnCode RestartEnvironment(object config, out string errorMessage)
        {
            try
            {
                terminal.Close();
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.GetType()}");
                Logger.Instance.Error($"{this} : {ex.Source}");
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
            terminal = null;
            try
            {
                process.Kill();
                process.Dispose();
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.GetType()}");
                Logger.Instance.Error($"{this} : {ex.Source}");
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
            process = null;
            
            var providerProcess = new Process();
            providerProcess.StartInfo.FileName = "python";
            providerProcess.StartInfo.Arguments = $"{ServerConfiguration.Instance.EnvironmentProviderPath} {ServerConfiguration.Instance.ServerAddress} {ServerConfiguration.Instance.ServerPort} {requesterGuid} {EnvironmentIndex}";
            providerProcess.Start();
            process = providerProcess;

            Launch(EnvironmentIndex, config, out errorMessage);
            return OperationReturnCode.Successiful;
        }

        public OperationReturnCode NeedRestart(out string errorMessage)
        {
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.NeedRestart,
                    parameters = new Dictionary<byte, object>()
                });
                errorMessage = "";
                return OperationReturnCode.Successiful;
            }
            else
            {
                errorMessage = "NotReady";
                return OperationReturnCode.NotExisted;
            }
        }
        public void NeedRestartResponse(bool result, string operationMessage)
        {
            EnvironmentRequester.NeedRestartResponse(EnvironmentIndex, result, operationMessage);
        }

        public bool Render(out string errorMessage)
        {
            if (IsReady)
            {
                SendRequest(new OperationRequestParameter
                {
                    operationCode = (byte)EnvironmentProviderOperationCode.Render,
                    parameters = new Dictionary<byte, object>()
                });
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = "NotReady";
                return false;
            }
        }
        public void RenderResponse(OperationReturnCode returnCode, object images, string operationMessage)
        {
            EnvironmentRequester.RenderResponse(returnCode, images, operationMessage);
        }
    }
}
