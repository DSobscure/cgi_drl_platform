using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace Cgi.VideoGame.Distributed.Server
{
    public class EnvironmentRequester
    {
        protected LocalPeer terminal;
        protected List<EnvironmentProvider> environmentProviders = new List<EnvironmentProvider>();

        private event Action<EnvironmentRequester> onClosed;
        public event Action<EnvironmentRequester> OnClosed { add { onClosed += value; } remove { onClosed -= value; } }

        public struct LaunchResult
        {
            public OperationReturnCode operationReturnCode;
            public object[] agentCounts;
            public string errorMessage;
            public int counter;
        }
        private LaunchResult launchResult;

        private event Action<LaunchResult> onLaunchFinished;
        public event Action<LaunchResult> OnLaunchFinished
        {
            add
            {
                lock (this)
                {
                    onLaunchFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onLaunchFinished -= value;
                }
            }
        }

        public struct GetActionSpaceResult
        {
            public OperationReturnCode operationReturnCode;
            public object actionSpace;
            public string errorMessage;
        }
        private GetActionSpaceResult getActionSpaceResult;

        private event Action<GetActionSpaceResult> onGetActionSpaceFinished;
        public event Action<GetActionSpaceResult> OnGetActionSpaceFinished
        {
            add
            {
                lock (this)
                {
                    onGetActionSpaceFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onGetActionSpaceFinished -= value;
                }
            }
        }

        public struct ResetAllResult
        {
            public OperationReturnCode operationReturnCode;
            public object[] observations;
            public string errorMessage;
            public int counter;
        }
        private ResetAllResult resetAllResult;

        private event Action<ResetAllResult> onResetAllFinished;
        public event Action<ResetAllResult> OnResetAllFinished
        {
            add
            {
                lock (this)
                {
                    onResetAllFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onResetAllFinished -= value;
                }
            }
        }

        public struct ResetResult
        {
            public OperationReturnCode operationReturnCode;
            public object observation;
            public string errorMessage;
        }
        private ResetResult resetResult;

        private event Action<ResetResult> onResetFinished;
        public event Action<ResetResult> OnResetFinished
        {
            add
            {
                lock (this)
                {
                    onResetFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onResetFinished -= value;
                }
            }
        }

        public struct GetTurnResult
        {
            public OperationReturnCode operationReturnCode;
            public object[] turns;
            public string errorMessage;
            public int counter;
        }
        private GetTurnResult getTurnResult;

        private event Action<GetTurnResult> onGetTurnFinished;
        public event Action<GetTurnResult> OnGetTurnFinished
        {
            add
            {
                lock (this)
                {
                    onGetTurnFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onGetTurnFinished -= value;
                }
            }
        }

        public struct SampleActionAllResult
        {
            public OperationReturnCode operationReturnCode;
            public object[] actions;
            public string errorMessage;
            public int counter;
        }
        private SampleActionAllResult sampleActionAllResult;

        private event Action<SampleActionAllResult> onSampleActionAllFinished;
        public event Action<SampleActionAllResult> OnSampleActionAllFinished
        {
            add
            {
                lock (this)
                {
                    onSampleActionAllFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onSampleActionAllFinished -= value;
                }
            }
        }

        public struct SampleActionResult
        {
            public OperationReturnCode operationReturnCode;
            public object action;
            public string errorMessage;
            public int counter;
        }
        private SampleActionResult sampleActionResult;

        private event Action<SampleActionResult> onSampleActionFinished;
        public event Action<SampleActionResult> OnSampleActionFinished
        {
            add
            {
                lock (this)
                {
                    onSampleActionFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onSampleActionFinished -= value;
                }
            }
        }

        public struct StepResult
        {
            public OperationReturnCode operationReturnCode;
            public object[] observations;
            public object[] rewards;
            public object[] dones;
            public object[] infos;
            public string errorMessage;
            public int counter;
        }
        private StepResult stepResult;

        private event Action<StepResult> onStepFinished;
        public event Action<StepResult> OnStepFinished
        {
            add
            {
                lock (this)
                {
                    onStepFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onStepFinished -= value;
                }
            }
        }

        public struct NeedRestartResult
        {
            public OperationReturnCode operationReturnCode;
            public bool[] result;
            public string errorMessage;
            public int counter;
        }
        private NeedRestartResult needRestartResult;

        private event Action<NeedRestartResult> onNeedRestartFinished;
        public event Action<NeedRestartResult> OnNeedRestartFinished
        {
            add
            {
                lock (this)
                {
                    onNeedRestartFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onNeedRestartFinished -= value;
                }
            }
        }

        public struct RenderResult
        {
            public OperationReturnCode operationReturnCode;
            public object images;
            public string errorMessage;
        }
        private RenderResult renderResult;

        private event Action<RenderResult> onRenderFinished;
        public event Action<RenderResult> OnRenderFinished
        {
            add
            {
                lock (this)
                {
                    onRenderFinished += value;
                }
            }
            remove
            {
                lock (this)
                {
                    onRenderFinished -= value;
                }
            }
        }


        internal EnvironmentRequester(LocalPeer terminal)
        {
            this.terminal = terminal;
            this.terminal.StartFastPooling();
            terminal.OnDisconnected += (t) =>
            {
                onClosed?.Invoke(this);
                Close();
            };
        }

        public void Close()
        {
            terminal.Close();
            foreach(var provider in environmentProviders)
            {
                provider.Close();
            }
        }

        public void SendResponse(OperationResponseParameter operationResponse)
        {
            terminal.SendResponse(new OperationResponseParameter
            {
                operationCode = (byte)OperationCode.EnvironmentRequesterRequest,
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
            terminal.SendRequest(new OperationRequestParameter
            {
                operationCode = (byte)OperationCode.EnvironmentRequesterRequest,
                parameters = new Dictionary<byte, object> {
                    { (byte)SubRequestParameterCode.SubRequestCode, operationRequest.operationCode },
                    { (byte)SubRequestParameterCode.SubRequestParameters, operationRequest.parameters }
                }
            });
        }

        public OperationReturnCode AllocateEnvironment(int environmentCount, out string message)
        {
            for (int i = 0; i < environmentCount; i++)
            {
                var providerProcess = new Process();
                providerProcess.StartInfo.FileName = "python";
                providerProcess.StartInfo.Arguments = $"{ServerConfiguration.Instance.EnvironmentProviderPath} {ServerConfiguration.Instance.ServerAddress} {ServerConfiguration.Instance.ServerPort} {terminal.Guid} {i}";
                providerProcess.Start();
                var provider = new EnvironmentProvider(i, terminal.Guid, providerProcess, this);
                environmentProviders.Add(provider);
            }
            Thread.Sleep(environmentCount * 100);
            while (true)
            {
                if (environmentProviders.All(x => x.IsReady))
                    break;
                Thread.Sleep(100);
            }
            message = "";
            return OperationReturnCode.Successiful;
        }

        public OperationReturnCode BindProviderTerminal(LocalPeer terminal, int environmentIndex, out string errorMessage)
        {
            lock (this)
            {
                return environmentProviders[environmentIndex].BindTerminal(terminal, out errorMessage);
            }
        }

        public OperationReturnCode Launch(object config, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                launchResult.operationReturnCode = OperationReturnCode.Successiful;
                launchResult.agentCounts = new object[environmentProviders.Count];
                launchResult.errorMessage = "";
                launchResult.counter = 0;

                for (int i = 0; i < environmentProviders.Count; i++)
                {
                    environmentProviders[i].Launch(i, config, out string subErrorMessage);
                    errorMessage += subErrorMessage;
                }
            }
            return OperationReturnCode.Successiful;   
        }

        public void LaunchResponse(int environmentIndex, OperationReturnCode returnCode, object agentCount, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    launchResult.operationReturnCode = returnCode;
                }
                launchResult.agentCounts[environmentIndex] = agentCount;
                launchResult.errorMessage += operationMessage;
                launchResult.counter += 1;
                if (launchResult.agentCounts.Length == launchResult.counter)
                {
                    onLaunchFinished?.Invoke(launchResult);
                    onLaunchFinished = null;
                }
                terminal.StartFastPooling();
            }
        }

        public OperationReturnCode GetActionSpace(int index, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                getActionSpaceResult.operationReturnCode = OperationReturnCode.Successiful;
                getActionSpaceResult.actionSpace = null;
                getActionSpaceResult.errorMessage = "";
                if(!environmentProviders[index].GetActionSpace(out string subErrorMessage))
                {
                    getActionSpaceResult.operationReturnCode = OperationReturnCode.NotExisted;
                }
                errorMessage += subErrorMessage;
            }
            return getActionSpaceResult.operationReturnCode;
        }
        public void GetActionSpaceResponse(OperationReturnCode returnCode, object actionSpace, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    getActionSpaceResult.operationReturnCode = returnCode;
                }
                getActionSpaceResult.actionSpace = actionSpace;
                getActionSpaceResult.errorMessage += operationMessage;
                onGetActionSpaceFinished?.Invoke(getActionSpaceResult);
                onGetActionSpaceFinished = null;
            }
        }

        public OperationReturnCode ResetAll(object resetSettings, out string errorMessage)
        {
            terminal.StartFastPooling();
            errorMessage = "";
            lock (this)
            {
                resetAllResult.operationReturnCode = OperationReturnCode.Successiful;
                resetAllResult.observations = new object[environmentProviders.Count];
                resetAllResult.errorMessage = "";
                resetAllResult.counter = 0;

                for (int i = 0; i < environmentProviders.Count; i++)
                {
                    if(!environmentProviders[i].Reset(resetSettings, out string subErrorMessage))
                    {
                        resetAllResult.counter += 1;
                    }
                    errorMessage += subErrorMessage;
                }
                if (resetAllResult.observations.Length == resetAllResult.counter)
                {
                    onResetAllFinished?.Invoke(resetAllResult);
                    onResetAllFinished = null;
                }
            }
            return OperationReturnCode.Successiful;
        }

        public void ResetAllResponse(int environmentIndex, OperationReturnCode returnCode, object observation, string operationMessage)
        {
            lock (this)
            {
                if (resetAllResult.observations == null)
                {
                    return;
                }
                if (returnCode != OperationReturnCode.Successiful)
                {
                    resetAllResult.operationReturnCode = returnCode;
                }
                resetAllResult.observations[environmentIndex] = observation;
                resetAllResult.errorMessage += operationMessage;
                resetAllResult.counter += 1;
                if (resetAllResult.observations.Length == resetAllResult.counter)
                {
                    onResetAllFinished?.Invoke(resetAllResult);
                    onResetAllFinished = null;
                }
            }
        }

        public OperationReturnCode Reset(int index, object resetSettings, out string errorMessage)
        {
            terminal.StartFastPooling();
            errorMessage = "";
            lock (this)
            {
                resetResult.operationReturnCode = OperationReturnCode.Successiful;
                resetResult.observation = null;
                resetResult.errorMessage = "";

                if(!environmentProviders[index].Reset(resetSettings, out string subErrorMessage))
                {
                    resetResult.operationReturnCode = OperationReturnCode.NotExisted;
                    onResetFinished?.Invoke(resetResult);
                    onResetFinished = null;
                }
                errorMessage += subErrorMessage;
            }
            return resetResult.operationReturnCode;
        }

        public void ResetResponse(OperationReturnCode returnCode, object observation, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    resetResult.operationReturnCode = returnCode;
                }
                resetResult.observation = observation;
                resetResult.errorMessage += operationMessage;

                onResetFinished?.Invoke(resetResult);
                onResetFinished = null;
            }
        }

        public OperationReturnCode GetTurn(out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                getTurnResult.operationReturnCode = OperationReturnCode.Successiful;
                getTurnResult.turns = new object[environmentProviders.Count];
                getTurnResult.errorMessage = "";
                getTurnResult.counter = 0;

                for (int i = 0; i < environmentProviders.Count; i++)
                {
                    if(!environmentProviders[i].GetTurn(out string subErrorMessage))
                    {
                        getTurnResult.counter += 1;
                    }
                    errorMessage += subErrorMessage;
                }
                if (getTurnResult.turns.Length == getTurnResult.counter)
                {
                    onGetTurnFinished?.Invoke(getTurnResult);
                    onGetTurnFinished = null;
                }
            }
            return OperationReturnCode.Successiful;
        }

        public void GetTurnResponse(int environmentIndex, OperationReturnCode returnCode, object turn, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    getTurnResult.operationReturnCode = returnCode;
                }
                getTurnResult.turns[environmentIndex] = turn;
                getTurnResult.errorMessage += operationMessage;
                getTurnResult.counter += 1;
                if (getTurnResult.turns.Length == getTurnResult.counter)
                {
                    onGetTurnFinished?.Invoke(getTurnResult);
                    onGetTurnFinished = null;
                }
            }
        }

        public OperationReturnCode SampleActionAll(out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                sampleActionAllResult.operationReturnCode = OperationReturnCode.Successiful;
                sampleActionAllResult.actions = new object[environmentProviders.Count];
                sampleActionAllResult.errorMessage = "";
                sampleActionAllResult.counter = 0;

                for (int i = 0; i < environmentProviders.Count; i++)
                {
                    if(!environmentProviders[i].SampleAction(out string subErrorMessage))
                    {
                        sampleActionAllResult.counter += 1;
                    }
                    errorMessage += subErrorMessage;
                }
                if (sampleActionAllResult.actions.Length == sampleActionAllResult.counter)
                {
                    onSampleActionAllFinished?.Invoke(sampleActionAllResult);
                    onSampleActionAllFinished = null;
                }
            }
            return OperationReturnCode.Successiful;
        }

        public void SampleActionAllResponse(int environmentIndex, OperationReturnCode returnCode, object action, string operationMessage)
        {
            lock (this)
            {
                if (sampleActionAllResult.actions == null)
                {
                    return;
                }
                if (returnCode != OperationReturnCode.Successiful)
                {
                    sampleActionAllResult.operationReturnCode = returnCode;
                }
                sampleActionAllResult.actions[environmentIndex] = action;
                sampleActionAllResult.errorMessage += operationMessage;
                sampleActionAllResult.counter += 1;
                if (sampleActionAllResult.actions.Length == sampleActionAllResult.counter)
                {
                    onSampleActionAllFinished?.Invoke(sampleActionAllResult);
                    onSampleActionAllFinished = null;
                }
            }
        }

        public OperationReturnCode SampleAction(int index, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                sampleActionResult.operationReturnCode = OperationReturnCode.Successiful;
                sampleActionResult.action = null;
                sampleActionResult.errorMessage = "";

                if(!environmentProviders[index].SampleAction(out string subErrorMessage))
                {
                    return OperationReturnCode.NotExisted;
                }
                errorMessage += subErrorMessage;
            }
            return OperationReturnCode.Successiful;
        }

        public void SampleActionResponse(OperationReturnCode returnCode, object action, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    sampleActionResult.operationReturnCode = returnCode;
                }
                sampleActionResult.action = action;
                sampleActionResult.errorMessage += operationMessage;

                onSampleActionFinished?.Invoke(sampleActionResult);
                onSampleActionFinished = null;
            }
        }

        public OperationReturnCode Step(object[] actionParameters, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                stepResult.operationReturnCode = OperationReturnCode.Successiful;
                stepResult.observations = new object[environmentProviders.Count];
                stepResult.rewards = new object[environmentProviders.Count];
                stepResult.dones = new object[environmentProviders.Count];
                stepResult.infos = new object[environmentProviders.Count];
                stepResult.errorMessage = "";
                stepResult.counter = 0;

                for (int i = 0; i < environmentProviders.Count; i++)
                {
                    if(!environmentProviders[i].Step(actionParameters[i], out string subErrorMessage))
                    {
                        stepResult.counter += 1;
                    }
                    errorMessage += subErrorMessage;
                }
                if (stepResult.observations.Length == stepResult.counter)
                {
                    onStepFinished?.Invoke(stepResult);
                    onStepFinished = null;
                }
            }
            return OperationReturnCode.Successiful;
        }

        public void StepResponse(int environmentIndex, OperationReturnCode returnCode, object observation, object reward, object done, object info, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    stepResult.operationReturnCode = returnCode;
                }
                stepResult.observations[environmentIndex] = observation;
                stepResult.rewards[environmentIndex] = reward;
                stepResult.dones[environmentIndex] = done;
                stepResult.infos[environmentIndex] = info;
                stepResult.errorMessage += operationMessage;
                stepResult.counter += 1;
                if (stepResult.observations.Length == stepResult.counter)
                {
                    onStepFinished?.Invoke(stepResult);
                    onStepFinished = null;
                }
            }
        }

        public OperationReturnCode RestartEnvironment(int index, object config, out string errorMessage)
        {
            return environmentProviders[index].RestartEnvironment(config, out errorMessage);
        }

        public OperationReturnCode NeedRestart(int index, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                if(index == -1)
                {
                    needRestartResult.operationReturnCode = OperationReturnCode.Successiful;
                    needRestartResult.result = new bool[environmentProviders.Count];
                    needRestartResult.errorMessage = "";
                    needRestartResult.counter = 0;

                    for (int i = 0; i < environmentProviders.Count; i++)
                    {
                        if (environmentProviders[i].NeedRestart(out string subErrorMessage) == OperationReturnCode.NotExisted)
                        {
                            needRestartResult.counter += 1;
                        }
                        errorMessage += subErrorMessage;
                    }
                }
                else
                {
                    needRestartResult.operationReturnCode = OperationReturnCode.Successiful;
                    needRestartResult.result = new bool[1];
                    needRestartResult.errorMessage = "";
                    needRestartResult.counter = 0;

                    if (environmentProviders[index].NeedRestart(out string subErrorMessage) == OperationReturnCode.NotExisted)
                    {
                        needRestartResult.counter += 1;
                    }
                    errorMessage += subErrorMessage;
                }
                if (needRestartResult.result.Length == needRestartResult.counter)
                {
                    onNeedRestartFinished?.Invoke(needRestartResult);
                    onNeedRestartFinished = null;
                }
                return needRestartResult.operationReturnCode;
            }
        }
        public void NeedRestartResponse(int environmentIndex, bool result, string operationMessage)
        {
            lock (this)
            {
                needRestartResult.result[environmentIndex] = result;
                needRestartResult.errorMessage += operationMessage;
                needRestartResult.counter += 1;
                if (needRestartResult.result.Length == needRestartResult.counter)
                {
                    onNeedRestartFinished?.Invoke(needRestartResult);
                    onNeedRestartFinished = null;
                }
            }
        }

        public OperationReturnCode Render(int index, out string errorMessage)
        {
            errorMessage = "";
            lock (this)
            {
                renderResult.operationReturnCode = OperationReturnCode.Successiful;
                renderResult.images = null;
                renderResult.errorMessage = "";

                if (!environmentProviders[index].Render(out string subErrorMessage))
                {
                    return OperationReturnCode.NotExisted;
                }
                errorMessage += subErrorMessage;
            }
            return OperationReturnCode.Successiful;
        }

        public void RenderResponse(OperationReturnCode returnCode, object images, string operationMessage)
        {
            lock (this)
            {
                if (returnCode != OperationReturnCode.Successiful)
                {
                    renderResult.operationReturnCode = returnCode;
                }
                renderResult.images = images;
                renderResult.errorMessage += operationMessage;

                onRenderFinished?.Invoke(renderResult);
                onRenderFinished = null;
            }
        }
    }
}