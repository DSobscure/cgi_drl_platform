using System;

namespace Cgi.VideoGame.Distributed.Server
{
    class Program
    {
        static void Main(string[] args)
        {
            ServerConfiguration.Load("Server.conf");
            Logger.Instance.SetLogLevel(ServerConfiguration.Instance.LogLevel);
            if (ServerConfiguration.Instance.IsWriteFile)
            {
                if (args.Length > 0)
                {
                    Logger.Instance.SetFilePath(args[0]);
                    Logger.Instance.System($"Write Log at {args[0]}");
                }
                else
                {
                    Logger.Instance.SetFilePath(ServerConfiguration.Instance.FilePath);
                }
            }

            if (ServerConfiguration.Instance.Version != "0.1.0")
            {
                Logger.Instance.Fatal($"Version error, current version 0.1.0, expected {ServerConfiguration.Instance.Version}");
            }
            else
            {
                var runtime = new ServerRuntime(ServerConfiguration.Instance.ServerPort);
            }
        }
    }
}
