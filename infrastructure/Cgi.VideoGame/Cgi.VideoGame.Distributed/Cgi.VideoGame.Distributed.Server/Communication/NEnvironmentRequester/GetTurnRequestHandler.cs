using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class GetTurnRequestHandler : EnvironmentRequesterRequestHandler
    {
        public GetTurnRequestHandler() : base(typeof(GetTurnRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnGetTurnFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)GetTurnResponseParameterCode.Turns, operationResult.turns }
                        }, operationResult.errorMessage);
                };

                OperationReturnCode returnCode = subject.GetTurn(out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
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
