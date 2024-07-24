import socket

class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"服务器监听在 {self.host}:{self.port}")

    def start(self):
        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                print(f"接受来自 {client_address} 的连接")
                try:
                    self.handle_client(client_socket)
                finally:
                    client_socket.close()
                    print(f"连接 {client_address} 已关闭")
        finally:
            self.stop()

    def handle_client(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print(f"收到数据: {data.decode('utf-8')}")
            with open('rate.txt', 'w', encoding='utf-8') as file:
                file.write(data.decode('utf-8'))
            with open('words.txt', 'w', encoding='utf-8') as file:
                file.write(data.decode('utf-8'))
            response = 'OK'
            client_socket.sendall(response.encode('utf-8'))

    def stop(self):
        self.server_socket.close()
        print("服务器已关闭")

if __name__ == "__main__":
    # 设置服务器地址和端口
    SERVER_HOST = '127.0.0.1'  # 使用0.0.0.0可以监听所有可用的网络接口
    SERVER_PORT = 12345        # 服务器端口

    # 创建服务器实例并启动
    server = TCPServer(SERVER_HOST, SERVER_PORT)
    server.start()
