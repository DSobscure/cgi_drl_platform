using Cgi.VideoGame.Distributed.Protocol;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    class LocalPeerOperationResponseRouter : OperationResponseRouter<LocalPeer, OperationCode>
    {
        public static LocalPeerOperationResponseRouter Instance { get; private set; } = new LocalPeerOperationResponseRouter();

        private LocalPeerOperationResponseRouter() : base("LocalPeer")
        {
            OperationTable.Add(OperationCode.EnvironmentProviderRequest, new EnvironmentProviderRequestResponseBroker());
            //OperationTable.Add(OperationCode.EnvironmentRequesterRequest, new EnvironmentRequesterRequestResponseBroker());
        }
    }
}
