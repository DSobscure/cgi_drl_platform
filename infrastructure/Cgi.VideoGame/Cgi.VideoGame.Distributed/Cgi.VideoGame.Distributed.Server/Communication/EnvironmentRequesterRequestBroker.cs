using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    class EnvironmentRequesterRequestBroker : OperationRequestHandler<LocalPeer, OperationCode>
    {
        protected Dictionary<EnvironmentRequesterOperationCode, EnvironmentRequesterRequestHandler> OperationTable { get; private set; } = new Dictionary<EnvironmentRequesterOperationCode, EnvironmentRequesterRequestHandler>();

        internal EnvironmentRequesterRequestBroker() : base(typeof(SubRequestParameterCode))
        {
            OperationTable.Add(EnvironmentRequesterOperationCode.AllocateEnvironment, new AllocateEnvironmentRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.Launch, new LaunchRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.GetActionSpace, new GetActionSpaceRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.ResetAll, new ResetAllRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.Reset, new ResetRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.GetTurn, new GetTurnRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.SampleActionAll, new SampleActionAllRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.SampleAction, new SampleActionRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.Step, new StepRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.RestartEnvironment, new RestartEnvironmentRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.NeedRestart, new NeedRestartRequestHandler());
            OperationTable.Add(EnvironmentRequesterOperationCode.Render, new RenderRequestHandler());
        }

        public override bool Handle(LocalPeer subject, OperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                EnvironmentRequesterOperationCode subRequestCode = (EnvironmentRequesterOperationCode)parameters[(byte)SubRequestParameterCode.SubRequestCode];
                Dictionary<byte, object> subRequestParameters = SerializationTool.Deserialize<Dictionary<byte, object>>((byte[])parameters[(byte)SubRequestParameterCode.SubRequestParameters]);
                if (OperationTable.ContainsKey(subRequestCode))
                {
                    EnvironmentRequester environmentRequester;
                    if (EnvironmentRequesterFactory.Instance.Find(subject.Guid, out environmentRequester))
                    {
                        if (OperationTable[subRequestCode].Handle(environmentRequester, subRequestCode, subRequestParameters, out errorMessage))
                        {
                            return true;
                        }
                        else
                        {
                            errorMessage = $"EnvironmentRequester-OperationRequest Error, SubOperationCode: {subRequestCode} from {subject}, HandlerErrorMessage: {errorMessage}";
                            return false;
                        }
                    }
                    else
                    {
                        errorMessage = $"EnvironmentRequester-OperationRequest Error, {subject} not in EnvironmentRequesterFactory";
                        return false;
                    }
                }
                else
                {
                    errorMessage = $"Unknow EnvironmentRequester-OperationRequest OperationCode:{operationCode} from {subject}";
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        public override void SendResponse(LocalPeer source, OperationCode operationCode, OperationReturnCode operationReturnCode, Dictionary<byte, object> parameters, string operationMessage)
        {
            source.SendResponse(new OperationResponseParameter
            {
                operationCode = (byte)operationCode,
                returnCode = (short)operationReturnCode,
                parameters = parameters,
                operationMessage = operationMessage
            });
            Logger.Instance.Info($"{source}[{operationCode}]:{operationMessage}");
        }
    }
}
