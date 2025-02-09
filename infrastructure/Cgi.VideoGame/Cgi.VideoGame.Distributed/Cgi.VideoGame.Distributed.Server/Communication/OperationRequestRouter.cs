using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    public abstract class OperationRequestRouter<TSubject, TOperationCode>
    {
        protected readonly string subjectName;
        protected Dictionary<TOperationCode, OperationRequestHandler<TSubject, TOperationCode>> OperationTable { get; private set; } = new Dictionary<TOperationCode, OperationRequestHandler<TSubject, TOperationCode>>();

        protected OperationRequestRouter(string subjectName)
        {
            this.subjectName = subjectName;
        }

        public bool Route(TSubject subject, TOperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (OperationTable.ContainsKey(operationCode))
            {
                if (OperationTable[operationCode].Handle(subject, operationCode, parameters, out errorMessage))
                {
                    return true;
                }
                else
                {
                    errorMessage = $"{subjectName}OperationRequest Error, OperationCode: {operationCode} from {subject}, HandlerErrorMessage: {errorMessage}";
                    return false;
                }
            }
            else
            {
                errorMessage = $"Unknow {subjectName}OperationRequest OperationCode:{operationCode} from {subject}";
                return false;
            }
        }
    }
}
