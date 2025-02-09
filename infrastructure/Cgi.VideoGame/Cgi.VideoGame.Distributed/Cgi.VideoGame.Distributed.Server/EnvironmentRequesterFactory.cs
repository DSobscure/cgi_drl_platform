using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server
{
    public class EnvironmentRequesterFactory
    {
        public static EnvironmentRequesterFactory Instance { get; } = new EnvironmentRequesterFactory();

        protected Dictionary<Guid, EnvironmentRequester> environmentRequesterTable = new Dictionary<Guid, EnvironmentRequester>();

        protected EnvironmentRequesterFactory()
        {

        }

        public OperationReturnCode Add(LocalPeer terminal, out string errorMessage)
        {
            lock (environmentRequesterTable)
            {
                if (environmentRequesterTable.ContainsKey(terminal.Guid))
                {
                    errorMessage = $"Terminal{terminal} already registered as an EnvironmentRequester!";
                    return OperationReturnCode.Duplicated;
                }
                else
                {
                    Logger.Instance.System("Create EnvironmentRequester");
                    environmentRequesterTable.Add(terminal.Guid, new EnvironmentRequester(terminal));
                    terminal.OnDisconnected += (t) => {
                        Remove(t.Guid);
                    };
                    errorMessage = "";
                    return OperationReturnCode.Successiful;
                }
            }
        }

        public void Remove(Guid terminalGuid)
        {
            lock (environmentRequesterTable)
            {
                if (environmentRequesterTable.ContainsKey(terminalGuid))
                {
                    environmentRequesterTable.Remove(terminalGuid);
                    Logger.Instance.System("Delete EnvironmentRequester");
                }
                else
                {
                    Logger.Instance.Error($"Delete{terminalGuid} Non-existed EnvironmentRequester");
                }
            }
        }

        public bool Find(Guid terminalGuid, out EnvironmentRequester environmentRequester)
        {
            lock (environmentRequesterTable)
            {
                if (environmentRequesterTable.ContainsKey(terminalGuid))
                {
                    environmentRequester = environmentRequesterTable[terminalGuid];
                    return true;
                }
                else
                {
                    Logger.Instance.Error($"Find{terminalGuid} Non-existed EnvironmentRequester");
                    environmentRequester = null;
                    return false;
                }
            }
        }
    }
}
