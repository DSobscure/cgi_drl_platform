using Cgi.VideoGame.Distributed.Protocol;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    class LocalPeerOperationRequestRouter : OperationRequestRouter<LocalPeer, OperationCode>
    {
        public static LocalPeerOperationRequestRouter Instance { get; private set; } = new LocalPeerOperationRequestRouter();

        private LocalPeerOperationRequestRouter() : base("LocalPeer")
        {
            OperationTable.Add(OperationCode.ReportIdentity, new ReportIdentityRequestHandler());
            //OperationTable.Add(OperationCode.EnvironmentProviderRequest, new EnvironmentProviderRequestBroker());
            OperationTable.Add(OperationCode.EnvironmentRequesterRequest, new EnvironmentRequesterRequestBroker());
        }
    }
}
