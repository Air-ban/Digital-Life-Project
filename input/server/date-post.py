import socket

# 设置服务器地址和端口
SERVER_HOST = '127.0.0.1'  # 使用0.0.0.0可以监听所有可用的网络接口
SERVER_PORT = 12345       # 服务器端口

# 创建一个TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定地址和端口
server_socket.bind((SERVER_HOST, SERVER_PORT))

# 监听连接
server_socket.listen(5)
print(f"服务器监听在 {SERVER_HOST}:{SERVER_PORT}")

try:
    while True:
        # 等待客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"接受来自 {client_address} 的连接")

        try:
            # 接收数据
            while True:
                data = client_socket.recv(1024)  # 设置接收数据的大小
                if not data:
                    break  # 如果没有数据，则断开连接
                print(f"收到数据: {data.decode('utf-8')}")
                with open('rate.txt', 'w', encoding='utf-8') as file:
                    file.write(data)
                with open('words.txt', 'w', encoding='utf-8') as file:
                    file.write(data)

               
                response = 'OK'
                client_socket.sendall(response.encode('utf-8'))

        finally:
            # 关闭客户端连接
            client_socket.close()
            print(f"连接 {client_address} 已关闭")
            break

finally:
    # 关闭服务器 socket
    server_socket.close()
    print("服务器已关闭")

# 欠完善，没有具体解析需求