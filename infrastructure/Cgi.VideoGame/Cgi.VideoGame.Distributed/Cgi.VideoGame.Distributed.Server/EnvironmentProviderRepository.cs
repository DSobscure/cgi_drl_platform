using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server
{
    public class EnvironmentProviderRepository
    {
        public static EnvironmentProviderRepository Instance { get; } = new EnvironmentProviderRepository();

        protected Dictionary<Guid, EnvironmentProvider> environmentProviderTable = new Dictionary<Guid, EnvironmentProvider>();

        protected EnvironmentProviderRepository()
        {

        }

        public OperationReturnCode Add(Guid terminalGuid, EnvironmentProvider provider, out string errorMessage)
        {
            lock (environmentProviderTable)
            {
                if (environmentProviderTable.ContainsKey(terminalGuid))
                {
                    errorMessage = $"Terminal{terminalGuid} already registered as an EnvironmentProvider!";
                    return OperationReturnCode.Duplicated;
                }
                else
                {
                    Logger.Instance.System("Create EnvironmentProvider");
                    environmentProviderTable.Add(terminalGuid, provider);
                    PeerFactory.Instance.FindPeer(terminalGuid, out LocalPeer peer);
                    peer.OnDisconnected += (t) => 
                    {
                        Remove(terminalGuid);
                    };
                    errorMessage = "";
                    return OperationReturnCode.Successiful;
                }
            }
        }

        public void Remove(Guid terminalGuid)
        {
            lock (environmentProviderTable)
            {
                if (environmentProviderTable.ContainsKey(terminalGuid))
                {
                    environmentProviderTable.Remove(terminalGuid);
                    Logger.Instance.System("Delete EnvironmentProvider");
                }
                else
                {
                    Logger.Instance.Error($"Delete{terminalGuid} Non-existed EnvironmentProvider");
                }
            }
        }

        public bool Find(Guid terminalGuid, out EnvironmentProvider environmentProvider)
        {
            lock (environmentProviderTable)
            {
                if (environmentProviderTable.ContainsKey(terminalGuid))
                {
                    environmentProvider = environmentProviderTable[terminalGuid];
                    return true;
                }
                else
                {
                    Logger.Instance.Error($"Find{terminalGuid} Non-existed EnvironmentProvider");
                    environmentProvider = null;
                    return false;
                }
            }
        }
    }
}
