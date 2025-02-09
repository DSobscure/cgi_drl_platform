using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace Cgi.VideoGame.Distributed.Server
{
    public class LocalPeer
    {
        protected Guid guid;
        public Guid Guid { get { return guid; } }
        TcpClient tcpClient;
        byte[] receiveBuffer = new byte[512000];
        int offset = 0;
        int remainedLength = 0;
        bool isCountingLength = true;
        byte[] receiveLengthHeader;
        int receiveLengthHeaderIndex;
        int pollingSpan = 1000;

        private event Action<LocalPeer> onDisconnected;
        public event Action<LocalPeer> OnDisconnected { add { onDisconnected += value; } remove { onDisconnected -= value; } }

        public LocalPeer(Guid guid, TcpClient tcpClient)
        {
            this.guid = guid;
            this.tcpClient = tcpClient;
            this.tcpClient.NoDelay = true;
            Task.Run(() => PeerMain());
        }

        public void StartFastPooling()
        {
            pollingSpan = 1;
        }

        void PeerMain()
        {
            try
            {
                OnDisconnected += (peer) => {
                    PeerFactory.Instance.DeletePeer(peer.Guid);
                };
                
                while (true)
                {
                    if (isCountingLength)
                    {
                        int oneByte = tcpClient.GetStream().ReadByte();
                        if (receiveLengthHeader == null)
                        {
                            receiveLengthHeader = new byte[oneByte];
                            receiveLengthHeaderIndex = 0;
                        }
                        else
                        {
                            receiveLengthHeader[receiveLengthHeaderIndex] = (byte)oneByte;
                            receiveLengthHeaderIndex++;
                            if (receiveLengthHeaderIndex == receiveLengthHeader.Length)
                            {
                                for (int i = 0; i < receiveLengthHeader.Length; i++)
                                {
                                    remainedLength += receiveLengthHeader[i] << (8 * i);
                                }
                                receiveLengthHeader = null;
                                isCountingLength = false;
                            }
                        }
                    }
                    else
                    {
                        int bytes = tcpClient.GetStream().Read(receiveBuffer, offset, remainedLength);
                        offset += bytes;
                        remainedLength -= bytes;
                        if (remainedLength == 0)
                        {
                            offset = 0;
                            isCountingLength = true;

                            var operationContent = SerializationTool.Deserialize<Dictionary<string, object>>(SerializationTool.Decompress(receiveBuffer));
                            switch (operationContent.Count)
                            {
                                case 2:
                                    OperationRequestParameter requestParameter = new OperationRequestParameter
                                    {
                                        operationCode = (byte)operationContent["operationCode"],
                                        parameters = SerializationTool.Deserialize<Dictionary<byte, object>>((byte[])operationContent["parameters"]),
                                    };
                                    OnOperationRequest(requestParameter);
                                    break;
                                case 4:
                                    OperationResponseParameter responseParameter = new OperationResponseParameter
                                    {
                                        operationCode = (byte)operationContent["operationCode"],
                                        returnCode = Convert.ToInt16(operationContent["returnCode"]),
                                        parameters = SerializationTool.Deserialize<Dictionary<byte, object>>((byte[])operationContent["parameters"]),
                                        operationMessage = (string)operationContent["operationMessage"]
                                    };
                                    OnOperationResponse(responseParameter);
                                    break;
                                default:
                                    Logger.Instance.Error($"Unknow Operation, Content Count: {operationContent.Count}");
                                    break;
                            }
                        }
                    }
                }
            }
            catch (ObjectDisposedException)
            {
                Logger.Instance.System($"{this} Disconnected");
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.GetType()}");
                Logger.Instance.Error($"{this} : {ex.Source}");
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
            onDisconnected?.Invoke(this);
            onDisconnected = null;
        }

        void Send(byte[] data)
        {
            int messageLength = data.Length;
            List<byte> lengthHeader = new List<byte>();
            while (messageLength > 255)
            {
                lengthHeader.Add((byte)(messageLength % 256));
                messageLength = messageLength / 256;
            }
            lengthHeader.Add((byte)messageLength);
            tcpClient.GetStream().WriteByte((byte)(lengthHeader.Count));
            foreach (var b in lengthHeader)
            {
                tcpClient.GetStream().WriteByte(b);
            }
            tcpClient.GetStream().Write(data, 0, data.Length);
        }
        protected void OnOperationRequest(OperationRequestParameter requestParameter)
        {
            string errorMessage;
            if (!Communication.LocalPeerOperationRequestRouter.Instance.Route(this, (OperationCode)requestParameter.operationCode, requestParameter.parameters, out errorMessage))
            {
                Logger.Instance.Error(errorMessage);
            }
        }

        protected void OnOperationResponse(OperationResponseParameter responseParameter)
        {
            string errorMessage;
            if (!Communication.LocalPeerOperationResponseRouter.Instance.Route(
                this, (OperationCode)responseParameter.operationCode,
                (OperationReturnCode)responseParameter.returnCode, responseParameter.parameters,
                responseParameter.operationMessage, out errorMessage))
            {
                Logger.Instance.Error(errorMessage);
            }
        }

        public void SendResponse(OperationResponseParameter operationResponse)
        {
            try
            {
                byte[] data = SerializationTool.Serialize(
                    new Dictionary<string, object> {
                        { "operationCode" , operationResponse.operationCode },
                        { "returnCode" , operationResponse.returnCode },
                        { "parameters" , SerializationTool.Serialize(operationResponse.parameters) },
                        { "operationMessage", operationResponse.operationMessage }
                    });
                Send(SerializationTool.Compress(data));
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
        }

        public void SendRequest(OperationRequestParameter operationRequest)
        {
            try
            {
                byte[] data = SerializationTool.Serialize(
                    new Dictionary<string, object> {
                        { "operationCode" , (byte)operationRequest.operationCode },
                        { "parameters" , SerializationTool.Serialize(operationRequest.parameters) },
                    });
                Send(SerializationTool.Compress(data));
            }
            catch (Exception ex)
            {
                Logger.Instance.Error($"{this} : {ex.Message}");
                Logger.Instance.Error($"{this} : {ex.StackTrace}");
            }
        }

        public override string ToString()
        {
            return $"Peer{guid}";
        }

        public void Close()
        {
            tcpClient.Close();
        }
    }
}
