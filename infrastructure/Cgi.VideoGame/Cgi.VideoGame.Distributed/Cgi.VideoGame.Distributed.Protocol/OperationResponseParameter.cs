using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Protocol
{
    public struct OperationResponseParameter
    {
        public byte operationCode;
        public short returnCode;
        public Dictionary<byte, object> parameters;
        public string operationMessage;
    }
}
