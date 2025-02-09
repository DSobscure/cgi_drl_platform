using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class ResetResponseHandler : EnvironmentProviderResponseHandler
    {
        public ResetResponseHandler() : base(typeof(ResetResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object observation = parameters[(byte)ResetResponseParameterCode.Observation];
                subject.ResetResponse(returnCode, observation, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
