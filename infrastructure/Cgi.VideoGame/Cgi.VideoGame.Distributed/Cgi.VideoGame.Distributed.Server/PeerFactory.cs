using System;
using System.Collections.Generic;
using System.Net.Sockets;

namespace Cgi.VideoGame.Distributed.Server
{
    class PeerFactory
    {
        public static PeerFactory Instance { get; } = new PeerFactory();

        private event Action<LocalPeer> onPeerDeleted;
        public event Action<LocalPeer> OnPeerDeleted { add { onPeerDeleted += value; } remove { onPeerDeleted -= value; } }

        private Dictionary<Guid, LocalPeer> peerDictionary = new Dictionary<Guid, LocalPeer>();


        public PeerFactory()
        {

        }

        public void CreatePeer(TcpClient client)
        {
            lock (peerDictionary)
            {
                Guid guid = Guid.NewGuid();
                LocalPeer peer = new LocalPeer(guid, client);
                peerDictionary.Add(guid, peer);
                Logger.Instance.System($"Peer {guid} created");
            }
        }

        public bool FindPeer(Guid guid, out LocalPeer peer)
        {
            lock (peerDictionary)
            {
                if (peerDictionary.ContainsKey(guid))
                {
                    peer = peerDictionary[guid];
                    return true;
                }
                else
                {
                    peer = null;
                    return false;
                }
            }
        }

        public bool DeletePeer(Guid guid)
        {
            lock (peerDictionary)
            {
                if (peerDictionary.ContainsKey(guid))
                {
                    LocalPeer peer = peerDictionary[guid];
                    onPeerDeleted?.Invoke(peer);
                    Logger.Instance.System($"Peer {peer} deleted");
                    return peerDictionary.Remove(guid);
                }
                else
                {
                    return false;
                }
            }
        }
    }
}
