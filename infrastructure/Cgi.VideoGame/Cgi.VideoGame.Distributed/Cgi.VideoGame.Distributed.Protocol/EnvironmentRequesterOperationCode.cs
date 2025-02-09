namespace Cgi.VideoGame.Distributed.Protocol
{
    public enum EnvironmentRequesterOperationCode : byte
    {
        AllocateEnvironment,
        Launch,
        GetActionSpace,
        ResetAll,
        Reset,
        GetTurn,
        SampleActionAll,
        SampleAction,
        Step,
        RestartEnvironment,
        NeedRestart,
        Render
    }
}
