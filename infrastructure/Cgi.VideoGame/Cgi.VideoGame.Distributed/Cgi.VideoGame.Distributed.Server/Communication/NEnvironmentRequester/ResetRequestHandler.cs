using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class ResetRequestHandler : EnvironmentRequesterRequestHandler
    {
        public ResetRequestHandler() : base(typeof(ResetRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnResetFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)ResetResponseParameterCode.Observation, operationResult.observation }
                        }, operationResult.errorMessage);
                };

                int index = Convert.ToInt32(parameters[(byte)ResetRequestParameterCode.Index]);
                object resetSettings = parameters[(byte)ResetRequestParameterCode.ResetSettings];
                OperationReturnCode returnCode = subject.Reset(index, resetSettings, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
                    return true;
                }
                else if (returnCode == OperationReturnCode.NotExisted)
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
