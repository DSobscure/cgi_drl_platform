using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Protocol
{
    public struct OperationRequestParameter
    {
        public byte operationCode;
        public Dictionary<byte, object> parameters;
    }
}
