using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class NeedRestartResponseHandler : EnvironmentProviderResponseHandler
    {
        public NeedRestartResponseHandler() : base(typeof(NeedRestartResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                bool result = Convert.ToBoolean(parameters[(byte)NeedRestartResponseParameterCode.Result]);
                subject.NeedRestartResponse(result, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
