import sys
sys.path.append('gen-py')

## 
##  Make sure to import the service and types 
##  defined in the thrift file
## 
from supernode import supernode
from supernode.ttypes import node
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def main():
    
    # read ip and port from command line
    if len(sys.argv) < 3:
        print("python3 supernode_client.py <supernode_ip> <supernode_port> <client_port>")
        return
    super_ip = sys.argv[1]
    super_port = int(sys.argv[2])
    client_port = int(sys.argv[3])

    # Thrift client boilerplate
    # Enjoy the enthusiastic comments included by the thrift developers

    # Make socket
    transport = TSocket.TSocket(super_ip, super_port)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = supernode.Client(protocol)

    # Connect!
    transport.open()
    
    id = client.request_join(client_port)
    print(id)
    if id != -1:
        if client.confirm_join():
            print(client.get_node())

    # Close!
    transport.close()


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)