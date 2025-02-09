using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class SampleActionRequestHandler : EnvironmentRequesterRequestHandler
    {
        public SampleActionRequestHandler() : base(typeof(SampleActionRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnSampleActionFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)SampleActionResponseParameterCode.Action, operationResult.action }
                        }, operationResult.errorMessage);
                };

                int index = Convert.ToInt32(parameters[(byte)SampleActionRequestParameterCode.Index]);
                OperationReturnCode returnCode = subject.SampleAction(index, out errorMessage);
                if (returnCode == OperationReturnCode.Successiful)
                {
                    return true;
                }
                else if (returnCode == OperationReturnCode.NotExisted)
                {
                    SendResponse(subject, operationCode, returnCode, new Dictionary<byte, object> {
                        { (byte)SampleActionResponseParameterCode.Action, null }
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
