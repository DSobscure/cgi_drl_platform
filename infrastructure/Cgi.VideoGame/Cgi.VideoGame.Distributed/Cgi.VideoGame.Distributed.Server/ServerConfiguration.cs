using System.IO;
using System.Xml;
using System.Xml.Serialization;

namespace Cgi.VideoGame.Distributed.Server
{
    public class ServerConfiguration
    {
        private static ServerConfiguration instance;
        public static ServerConfiguration Instance { get { return instance; } }

        [XmlElement]
        public string Version { get; set; }
        [XmlElement]
        public string ServerAddress { get; set; }
        [XmlElement]
        public int ServerPort { get; set; }
        [XmlElement]
        public LogLevel LogLevel { get; set; }
        [XmlElement]
        public bool IsWriteFile { get; set; }
        [XmlElement]
        public string FilePath { get; set; }
        [XmlElement]
        public string EnvironmentProviderPath { get; set; }

        public ServerConfiguration() { }
        public static void Load(string filePath)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(ServerConfiguration));
            if (File.Exists(filePath))
            {
                using (XmlReader reader = XmlReader.Create(filePath))
                {
                    if (serializer.CanDeserialize(reader))
                    {
                        instance = (ServerConfiguration)serializer.Deserialize(reader);
                    }
                    else
                    {
                        instance = null;
                        Logger.Instance.Fatal("server configuration can't be serialized!");
                    }
                }
            }
            else
            {
                instance = new ServerConfiguration
                {
                    Version = "0.1.0",
                    ServerAddress = "127.0.0.1",
                    ServerPort = 30001,
                    LogLevel = LogLevel.All,
                    IsWriteFile = true,
                    FilePath = "log.txt",
                    EnvironmentProviderPath = "environment_provider.py"
                };
                using (XmlWriter writer = XmlWriter.Create(filePath))
                {
                    serializer.Serialize(writer, instance);
                }
            }
        }
    }
}
