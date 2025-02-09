using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    class EnvironmentProviderRequestResponseBroker : OperationResponseHandler<LocalPeer, OperationCode>
    {
        protected Dictionary<EnvironmentProviderOperationCode, EnvironmentProviderResponseHandler> OperationTable { get; private set; } = new Dictionary<EnvironmentProviderOperationCode, EnvironmentProviderResponseHandler>();

        internal EnvironmentProviderRequestResponseBroker() : base(typeof(SubRequestResponseParameterCode))
        {
            OperationTable.Add(EnvironmentProviderOperationCode.Launch, new LaunchResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.GetActionSpace, new GetActionSpaceResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.Reset, new ResetResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.GetTurn, new GetTurnResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.SampleAction, new SampleActionResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.Step, new StepResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.ServerLaunch, new ServerLaunchResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.NeedRestart, new NeedRestartResponseHandler());
            OperationTable.Add(EnvironmentProviderOperationCode.Render, new RenderResponseHandler());
        }

        public override bool Handle(LocalPeer subject, OperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                EnvironmentProviderOperationCode subRequestCode = (EnvironmentProviderOperationCode)parameters[(byte)SubRequestResponseParameterCode.SubRequestCode];
                OperationReturnCode subReturnCode = (OperationReturnCode)Convert.ToInt16(parameters[(byte)SubRequestResponseParameterCode.SubRequestReturnCode]);
                Dictionary<byte, object> subRequestParameters = SerializationTool.Deserialize<Dictionary<byte, object>>((byte[])parameters[(byte)SubRequestResponseParameterCode.SubRequestResponseParameters]);
                string subRequestOperationMessage = (string)parameters[(byte)SubRequestResponseParameterCode.SubRequestOperationMessage];

                if (OperationTable.ContainsKey(subRequestCode))
                {
                    if (EnvironmentProviderRepository.Instance.Find(subject.Guid, out EnvironmentProvider environmentProvider))
                    {
                        if (OperationTable[subRequestCode].Handle(environmentProvider, subRequestCode, subReturnCode, subRequestParameters, subRequestOperationMessage, out errorMessage))
                        {
                            return true;
                        }
                        else
                        {
                            errorMessage = $"EnvironmentProvider-OperationResponse Error, SubOperationCode: {subRequestCode} from {subject}, HandlerErrorMessage: {errorMessage}";
                            return false;
                        }
                    }
                    else
                    {
                        errorMessage = $"EnvironmentProvider-OperationResponse Error, {subject} not in EnvironmentProviderFactory";
                        return false;
                    }
                }
                else
                {
                    errorMessage = $"Unknow EnvironmentProvider-OperationResponse OperationCode:{operationCode} from {subject} OperationMessage: {operationMessage}";
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
    }
}
