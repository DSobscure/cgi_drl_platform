using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class AllocateEnvironmentRequestHandler : EnvironmentRequesterRequestHandler
    {
        public AllocateEnvironmentRequestHandler() : base(typeof(AllocateEnvironmentRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                int environmentCount = Convert.ToInt32(parameters[(byte)AllocateEnvironmentRequestParameterCode.EnvironmentCount]);
                OperationReturnCode returnCode = subject.AllocateEnvironment(environmentCount, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
                    SendResponse(subject, operationCode, returnCode, new Dictionary<byte, object>(), errorMessage);
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
