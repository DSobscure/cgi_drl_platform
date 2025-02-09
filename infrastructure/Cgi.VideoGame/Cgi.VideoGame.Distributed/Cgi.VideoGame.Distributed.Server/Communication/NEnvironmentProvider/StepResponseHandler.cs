using Cgi.VideoGame.Distributed.Protocol;
using Cgi.VideoGame.Distributed.Protocol.NEnvironmentProvider;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication.NEnvironmentProvider
{
    class StepResponseHandler : EnvironmentProviderResponseHandler
    {
        public StepResponseHandler() : base(typeof(StepResponseParameterCode))
        {
        }

        public override bool Handle(EnvironmentProvider subject, EnvironmentProviderOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (base.Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
            {
                object observation = parameters[(byte)StepResponseParameterCode.Observation];
                object reward = parameters[(byte)StepResponseParameterCode.Reward];
                object done = parameters[(byte)StepResponseParameterCode.Done];
                object info = parameters[(byte)StepResponseParameterCode.Info];
                subject.StepResponse(returnCode, observation, reward, done, info, operationMessage);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
