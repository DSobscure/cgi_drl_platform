using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    abstract class EnvironmentRequesterRequestHandler : OperationRequestHandler<EnvironmentRequester, EnvironmentRequesterOperationCode>
    {
        public EnvironmentRequesterRequestHandler(Type typeOfOperationRequestParameterCode) : base(typeOfOperationRequestParameterCode)
        {
        }

        public override void SendResponse(EnvironmentRequester source, EnvironmentRequesterOperationCode operationCode, OperationReturnCode operationReturnCode, Dictionary<byte, object> parameters, string operationMessage)
        {
            source.SendResponse(new OperationResponseParameter
            {
                operationCode = (byte)operationCode,
                returnCode = (short)operationReturnCode,
                parameters = parameters,
                operationMessage = operationMessage
            });
            Logger.Instance.Info($"{source}[{operationCode}]:{operationMessage}");
        }
    }
}
