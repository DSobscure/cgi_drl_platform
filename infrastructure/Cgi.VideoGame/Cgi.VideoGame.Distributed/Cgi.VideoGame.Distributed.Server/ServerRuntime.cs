using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace Cgi.VideoGame.Distributed.Server
{
    class ServerRuntime
    {
        private int port;
        private TcpListener serverListener;

        public ServerRuntime(int port)
        {
            this.port = port;
            var serverIP = IPAddress.Parse(ServerConfiguration.Instance.ServerAddress);
            serverListener = new TcpListener(serverIP, port);
            serverListener.Start();
            Logger.Instance.System($"Start listening at port {port}");
            AcceptConnectionLoop();
        }

        void AcceptConnectionLoop()
        {
            Logger.Instance.System("Waiting for connection ....");
            while (true)
            {
                TcpClient client = serverListener.AcceptTcpClient();
                Logger.Instance.System($"Accept connectiion!");
                PeerFactory.Instance.CreatePeer(client);
                Thread.Sleep(100);
            }
        }
    }
}
