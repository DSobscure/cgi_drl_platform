using Cgi.VideoGame.Distributed.Protocol;
using System;
using System.Collections.Generic;

namespace Cgi.VideoGame.Distributed.Server.Communication
{
    class ReportIdentityRequestHandler : LocalPeerRequestHandler
    {
        public ReportIdentityRequestHandler() : base(typeof(ReportIdentityRequestParameterCode))
        {
        }

        public override bool Handle(LocalPeer terminal, OperationCode operationCode, Dictionary<byte, object> parameters, out string errorMessage)
        {
            if (base.Handle(terminal, operationCode, parameters, out errorMessage))
            {
                IdentityCode identityCode = (IdentityCode)(byte)parameters[(byte)ReportIdentityRequestParameterCode.Identity];
                OperationReturnCode returnCode;
                switch (identityCode)
                {
                    case IdentityCode.EnvironmentProvider:
                        Guid requesterGuid = Guid.Parse((string)parameters[(byte)ReportIdentityRequestParameterCode.RequesterGuid]);
                        int environmentIndex = Convert.ToInt32((string)parameters[(byte)ReportIdentityRequestParameterCode.EnvironmentIndex]);
                        EnvironmentRequester requester;
                        if (EnvironmentRequesterFactory.Instance.Find(requesterGuid, out requester))
                            returnCode = requester.BindProviderTerminal(terminal, environmentIndex, out errorMessage);
                        else
                            returnCode = OperationReturnCode.NotExisted;
                        break;
                    case IdentityCode.EnvironmentRequester:
                        returnCode = EnvironmentRequesterFactory.Instance.Add(terminal, out errorMessage);
                        break;
                    default:
                        returnCode = OperationReturnCode.UndefinedError;
                        errorMessage = $"ReportIdentity IdentityCode: {identityCode}";
                        break;
                }
                return returnCode == OperationReturnCode.Successiful;
            }
            else
            {
                return false;
            }
        }
    }
}
