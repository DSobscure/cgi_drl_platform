using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    public abstract class OperationResponseHandler<TSubject, TOperationCode>
    {
        protected int CorrectParameterCount { get; private set; }

        protected OperationResponseHandler(Type typeOfOperationResponseParameterCode)
        {
            CorrectParameterCount = Enum.GetNames(typeOfOperationResponseParameterCode).Length;
        }

        public virtual bool Handle(TSubject subject, TOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            return CheckOperationReturn(returnCode, parameters, operationMessage, out errorMessage);
        }
        protected virtual bool CheckOperationReturn(OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            switch (returnCode)
            {
                case OperationReturnCode.Successiful:
                    return CheckParameters(parameters, out errorMessage);
                default:
                    errorMessage = $"Unknown OperationReturnCode: {returnCode}, OperationMessage: {operationMessage}";
                    return false;
            }
        }
        protected virtual bool CheckParameters(Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (parameters.Count != CorrectParameterCount)
            {
                errorMessage = $"Parameter Count: {parameters.Count} Should be {CorrectParameterCount}";
                return false;
            }
            else
            {
                errorMessage = "";
                return true;
            }
        }
    }
}
