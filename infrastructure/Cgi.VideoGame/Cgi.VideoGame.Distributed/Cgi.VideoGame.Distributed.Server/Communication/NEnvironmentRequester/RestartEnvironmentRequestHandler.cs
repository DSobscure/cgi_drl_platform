using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class RestartEnvironmentRequestHandler : EnvironmentRequesterRequestHandler
    {
        public RestartEnvironmentRequestHandler() : base(typeof(RestartEnvironmentRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                int index = Convert.ToInt32(parameters[(byte)RestartEnvironmentRequestParameterCode.Index]);
                object config = parameters[(byte)RestartEnvironmentRequestParameterCode.Config];
                OperationReturnCode returnCode = subject.RestartEnvironment(index, config, out errorMessage);
                return returnCode == OperationReturnCode.Successiful;
            }
            else
            {
                return false;
            }
        }
    }
}
