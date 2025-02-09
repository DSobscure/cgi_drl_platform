using Cgi.VideoGame.Distributed.Protocol;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    public abstract class OperationResponseRouter<TSubject, TOperationCode>
    {
        private string subjectName;
        protected Dictionary<TOperationCode, OperationResponseHandler<TSubject, TOperationCode>> OperationTable { get; private set; } = new Dictionary<TOperationCode, OperationResponseHandler<TSubject, TOperationCode>>();

        protected OperationResponseRouter(string subjectName)
        {
            this.subjectName = subjectName;
        }

        public bool Route(TSubject subject, TOperationCode operationCode, OperationReturnCode returnCode, Dictionary<byte, object> parameters, string operationMessage, out string errorMessage)
        {
            if (OperationTable.ContainsKey(operationCode))
            {
                if (OperationTable[operationCode].Handle(subject, operationCode, returnCode, parameters, operationMessage, out errorMessage))
                {
                    return true;
                }
                else
                {
                    errorMessage = $"{subjectName}OperationResponse Error OperationCode:{operationCode} HandlerErrorMessage: {errorMessage}";
                    return false;
                }
            }
            else
            {
                errorMessage = $"Unknow {subjectName}OperationResponse OperationCode:{operationCode}";
                return false;
            }
        }
    }
}
