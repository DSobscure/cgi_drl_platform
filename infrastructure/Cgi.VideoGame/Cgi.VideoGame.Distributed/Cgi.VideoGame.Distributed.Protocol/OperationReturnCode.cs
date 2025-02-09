namespace Cgi.VideoGame.Distributed.Protocol
{
    public enum OperationReturnCode : short
    {
        UndefinedError,
        Successiful,
        ParameterCountError,
        NullObject,
        DbTransactionFailed,
        DbNoChanged,
        ParameterFormateError,
        Duplicated,
        NotExisted,
        AuthenticationFailed,
        InstantiateFailed,
        OutOfResource,
    }
}
