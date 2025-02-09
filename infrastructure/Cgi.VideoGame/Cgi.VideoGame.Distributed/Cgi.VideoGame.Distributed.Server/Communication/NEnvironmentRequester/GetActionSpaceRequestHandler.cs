using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class GetActionSpaceRequestHandler : EnvironmentRequesterRequestHandler
    {
        public GetActionSpaceRequestHandler() : base(typeof(GetActionSpaceRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnGetActionSpaceFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)GetActionSpaceResponseParameterCode.ActionSpace, operationResult.actionSpace }
                        }, operationResult.errorMessage);
                };

                int index = Convert.ToInt32(parameters[(byte)GetActionSpaceRequestParameterCode.Index]);
                OperationReturnCode returnCode = subject.GetActionSpace(index, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
                    return true;
                }
                else if(returnCode == OperationReturnCode.NotExisted)
                {
                    SendResponse(subject, operationCode, returnCode, new Dictionary<byte, object> {
                        { (byte)GetActionSpaceResponseParameterCode.ActionSpace, null }
                    }, errorMessage);
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
