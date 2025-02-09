using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class LaunchRequestHandler : EnvironmentRequesterRequestHandler
    {
        public LaunchRequestHandler() : base(typeof(LaunchRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnLaunchFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)LaunchResponseParameterCode.AgentCounts, operationResult.agentCounts }
                        }, operationResult.errorMessage);
                };

                object config = parameters[(byte)LaunchRequestParameterCode.Config];
                OperationReturnCode returnCode = subject.Launch(config, out errorMessage);
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
