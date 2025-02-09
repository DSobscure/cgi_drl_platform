using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentRequester;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentRequester
{
    class StepRequestHandler : EnvironmentRequesterRequestHandler
    {
        public StepRequestHandler() : base(typeof(StepRequestParameterCode))
        {
        }

        public override bool Handle(EnvironmentRequester subject, EnvironmentRequesterOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, parameters, out errorMessage))
            {
                subject.OnStepFinished += (operationResult) => {
                    SendResponse(subject, operationCode, operationResult.operationReturnCode, new Dictionary<byte, object> {
                            { (byte)StepResponseParameterCode.Observations, operationResult.observations },
                            { (byte)StepResponseParameterCode.Rewards, operationResult.rewards },
                            { (byte)StepResponseParameterCode.Dones, operationResult.dones },
                            { (byte)StepResponseParameterCode.Infos, operationResult.infos }
                        }, operationResult.errorMessage);
                };

                object[] actionParameters = (object[])parameters[(byte)StepRequestParameterCode.ActionParameters];
                OperationReturnCode returnCode = subject.Step(actionParameters, out errorMessage);
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
