using System;
using System.IO;

namespace Cgi.VideoGame.Distributed.Server
{
    [Flags]
    public enum LogLevel
    {
        None = 0,
        Info = 1,
        Warning = 2,
        Error = 4,
        Fatal = 8,
        Debug = 16,
        System = 32,
        All = None | Info | Warning | Error | Fatal | Debug | System,
        Important = Warning | Error | Fatal | System,
    }

    class Logger : IDisposable
    {
        public static Logger Instance { get; protected set; } = new Logger();

        private LogLevel loglLevel = LogLevel.All;
        protected object loggerLock = new object();
        protected TextWriter logWriter = null;

        protected ConsoleColor backgroundColor;
        protected ConsoleColor foregroundColor;

        protected Logger()
        {

        }


        public void SetLogLevel(LogLevel level)
        {
            lock (loggerLock)
            {
                loglLevel = level;
            }
        }

        public void SetFilePath(string filePath)
        {
            logWriter = File.AppendText(filePath);
        }

        protected void WriteLine(string message, ConsoleColor foregroundColor, ConsoleColor backgroundColor)
        {
            lock (loggerLock)
            {
                this.backgroundColor = Console.BackgroundColor;
                this.foregroundColor = Console.ForegroundColor;
                Console.ForegroundColor = foregroundColor;
                Console.BackgroundColor = backgroundColor;
                Console.WriteLine(message);
                logWriter?.WriteLine(message);
                logWriter?.Flush();
                Console.BackgroundColor = this.backgroundColor;
                Console.ForegroundColor = this.foregroundColor;
            }
        }
        protected void Write(string message, ConsoleColor foregroundColor, ConsoleColor backgroundColor)
        {
            this.backgroundColor = Console.BackgroundColor;
            this.foregroundColor = Console.ForegroundColor;
            Console.ForegroundColor = foregroundColor;
            Console.BackgroundColor = backgroundColor;
            Console.Write(message);
            logWriter?.Write(message);
            logWriter?.Flush();
            Console.BackgroundColor = this.backgroundColor;
            Console.ForegroundColor = this.foregroundColor;
        }

        public void Info(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.Info) != LogLevel.None)
                {
                    WriteLine($"[{DateTime.Now.ToString("o")}] Info: {message}", ConsoleColor.White, ConsoleColor.Black);
                }
            }
        }

        public void Warning(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.Warning) != LogLevel.None)
                {
                    WriteLine($"[{DateTime.Now.ToString("o")}] Warning: {message}", ConsoleColor.Yellow, ConsoleColor.Black);
                }
            }
        }

        public void Error(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.Error) != LogLevel.None)
                {
                    WriteLine($"[{DateTime.Now.ToString("o")}] Error: {message}", ConsoleColor.Red, ConsoleColor.Black);
                }
            }
        }

        public void Fatal(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.Fatal) != LogLevel.None)
                {
                    Write($"[{DateTime.Now.ToString("o")}] ", ConsoleColor.Red, ConsoleColor.Black);
                    WriteLine($"Fatal: {message}", ConsoleColor.Black, ConsoleColor.Red);
                }
            }
        }

        public void Debug(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.Debug) != LogLevel.None)
                {
                    WriteLine($"[{DateTime.Now.ToString("o")}] Debug: {message}", ConsoleColor.Green, ConsoleColor.Black);
                }
            }
        }

        public void System(string message)
        {
            lock (loggerLock)
            {
                if ((loglLevel & LogLevel.System) != LogLevel.None)
                {
                    WriteLine($"[{DateTime.Now.ToString("o")}] System: {message}", ConsoleColor.White, ConsoleColor.Black);
                }
            }
        }

        public void Dispose()
        {
            logWriter?.Close();
        }
    }
}
