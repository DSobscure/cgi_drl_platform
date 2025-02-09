using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class GetActionSpaceResponseHandler : EnvironmentProviderResponseHandler
    {
        public GetActionSpaceResponseHandler() : base(typeof(GetActionSpaceResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object actionSpace = parameters[(byte)GetActionSpaceResponseParameterCode.ActionSpace];
                subject.GetActionSpaceResponse(returnCode, actionSpace, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
