using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class GetTurnResponseHandler : EnvironmentProviderResponseHandler
    {
        public GetTurnResponseHandler() : base(typeof(GetTurnResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object turn = parameters[(byte)GetTurnResponseParameterCode.Turn];
                subject.GetTurnResponse(returnCode, turn, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
