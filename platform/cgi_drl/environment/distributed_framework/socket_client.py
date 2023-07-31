import socket
import msgpack
from cgi_drl.environment.distributed_framework.protocol import OperationCode, OperationReturnCode
import gzip
import msgpack_numpy as m
m.patch()
import time

class SocketClient():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server.connect((self.ip, self.port))  

    def _send(self, message):
        message = gzip.compress(message)
        length = len(message)
        length_header = []
        while length > 255:
            length_header.append(length%256)
            length = length // 256
        length_header.append(length)
        self.server.sendall(bytearray([len(length_header)]))
        self.server.sendall(bytearray(length_header))
        self.server.sendall(message)

    def _receive(self):
        length = 0
        byte = self.server.recv(1)
        if len(byte) == 0:
            print("Connection closed")
            exit(0)
        length_header_size = ord(byte)
        for i in range(length_header_size):
            oneByte = ord(self.server.recv(1))
            length += oneByte << (8 * i)
        recv_len = 0
        recv_buf = None
        while recv_len != length:
            tmp_buf = self.server.recv(length-recv_len)
            if len(tmp_buf) == 0:
                return None
            if recv_buf is None:
                recv_buf = tmp_buf
            else:
                recv_buf += tmp_buf
            recv_len += len(tmp_buf)
        recv_buf = gzip.decompress(recv_buf)
        return recv_buf

    def _send_operation_request(self, operation_code, parameters):
        request = msgpack.packb({
            "operationCode" : operation_code,
            "parameters" : msgpack.packb(parameters, use_bin_type=True)
        }, use_bin_type=True)
        self._send(request)

    def _receive_operation_response(self):
        message = self._receive()
        if message != None:
            response = msgpack.unpackb(message, raw=False, strict_map_key=False)
            operationCode = OperationCode(response["operationCode"])
            returnCode = OperationReturnCode(response["returnCode"])
            parameters = msgpack.unpackb(response["parameters"], raw=False, strict_map_key=False)
            operationMessage = str(response["operationMessage"])
            return operationCode, returnCode, parameters, operationMessage
        else:
            return None, None, None, None

    def _receive_operation_request(self):
        message = self._receive()
        if message != None:
            request = msgpack.unpackb(message, raw=False, strict_map_key=False)
            operationCode = request["operationCode"]
            parameters = msgpack.unpackb(request["parameters"], raw=False, strict_map_key=False)
            return operationCode, parameters
        else:
            return None, None

    def _send_operation_response(self, operation_code, return_code, parameters, operation_message):
        response = msgpack.packb({
            "operationCode" : operation_code,
            "returnCode" : return_code,
            "parameters" : msgpack.packb(parameters, use_bin_type=True),
            "operationMessage" : operation_message
        }, use_bin_type=True)
        self._send(response)

    def close(self):
        self.server.close()