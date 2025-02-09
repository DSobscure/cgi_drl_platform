using Cgi.VideoGame.Distributed.Protocol;
using System;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    abstract class EnvironmentProviderResponseHandler : OperationResponseHandler<EnvironmentProvider, EnvironmentProviderOperationCode>
    {
        public EnvironmentProviderResponseHandler(Type typeOfOperationResponseParameterCode) : base(typeOfOperationResponseParameterCode)
        {
        }
    }
}
