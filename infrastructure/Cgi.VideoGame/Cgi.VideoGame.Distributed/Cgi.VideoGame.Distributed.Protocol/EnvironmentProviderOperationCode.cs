namespace Cgi.VideoGame.Distributed.Protocol
{
    public enum EnvironmentProviderOperationCode : byte
    {
        Launch,
        GetActionSpace,
        Reset,
        GetTurn,
        SampleAction,
        Step,
        ServerLaunch,
        NeedRestart,
        Render
    }
}
