using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class ServerLaunchResponseHandler : EnvironmentProviderResponseHandler
    {
        public ServerLaunchResponseHandler() : base(typeof(ServerLaunchResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object agentCount = parameters[(byte)ServerLaunchResponseParameterCode.AgentCount];
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
