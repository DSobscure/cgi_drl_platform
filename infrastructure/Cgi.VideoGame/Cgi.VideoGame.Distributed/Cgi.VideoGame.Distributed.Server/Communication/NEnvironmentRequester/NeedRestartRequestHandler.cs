using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class NeedRestartRequestHandler : EnvironmentRequesterRequestHandler
    {
        public NeedRestartRequestHandler() : base(typeof(NeedRestartRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnNeedRestartFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)NeedRestartResponseParameterCode.Result, operationResult.result }
                        }, "");
                };

                int index = Convert.ToInt32(parameters[(byte)NeedRestartRequestParameterCode.Index]);
                OperationReturnCode returnCode = subject.NeedRestart(index, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful || returnCode == OperationReturnCode.NotExisted)
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
