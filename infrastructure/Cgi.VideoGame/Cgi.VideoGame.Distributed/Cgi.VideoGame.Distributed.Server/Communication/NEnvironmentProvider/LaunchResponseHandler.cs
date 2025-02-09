using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class LaunchResponseHandler : EnvironmentProviderResponseHandler
    {
        public LaunchResponseHandler() : base(typeof(LaunchResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object agentCount = parameters[(byte)LaunchResponseParameterCode.AgentCount];
                subject.LaunchResponse(returnCode, agentCount, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
