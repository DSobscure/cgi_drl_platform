using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class ResetAllRequestHandler : EnvironmentRequesterRequestHandler
    {
        public ResetAllRequestHandler() : base(typeof(ResetAllRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnResetAllFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)ResetAllResponseParameterCode.Observations, operationResult.observations }
                        }, operationResult.errorMessage);
                };

                object resetSettings = parameters[(byte)ResetAllRequestParameterCode.ResetSettings];
                OperationReturnCode returnCode = subject.ResetAll(resetSettings, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
                    return true;
                }
                else
                {
                    SendResponse(subject, operationCode, returnCode, new Dictionary<byte, object>(), errorMessage);
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
