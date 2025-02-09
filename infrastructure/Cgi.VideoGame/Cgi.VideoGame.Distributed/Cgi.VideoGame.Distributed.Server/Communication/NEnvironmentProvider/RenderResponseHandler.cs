using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class RenderResponseHandler : EnvironmentProviderResponseHandler
    {
        public RenderResponseHandler() : base(typeof(RenderResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object images = parameters[(byte)RenderResponseParameterCode.Images];
                subject.RenderResponse(returnCode, images, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
