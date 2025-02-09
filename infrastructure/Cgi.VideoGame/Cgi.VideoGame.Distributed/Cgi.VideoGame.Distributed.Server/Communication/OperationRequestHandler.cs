using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    public abstract class OperationRequestHandler<TSubject, TOperationCode>
    {
        protected int CorrectParameterCount { get; private set; }

        public OperationRequestHandler(Type typeOfOperationRequestParameterCode)
        {
            CorrectParameterCount = Enum.GetNames(typeOfOperationRequestParameterCode).Length;
        }

        public abstract void SendResponse(TSubject source, TOperationCode operationCode, OperationReturnCode operationReturnCode, Dictionary<byte, object> parameters, string operationMessage);

        public virtual bool Handle(TSubject subject, TOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (CheckParameterCount(parameters, out errorMessage))
            {
                return true;
            }
            else
            {
                SendResponse(subject, operationCode, OperationReturnCode.ParameterCountError, new Dictionary<byte, object>(), errorMessage);
                return false;
            }
        }
        private bool CheckParameterCount(Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (parameters.Count == CorrectParameterCount)
            {
                errorMessage = "";
                return true;
            }
            else
            {
                errorMessage = $"Parameter Count: {parameters.Count}, should be {CorrectParameterCount}";
                return false;
            }
        }
    }
}
