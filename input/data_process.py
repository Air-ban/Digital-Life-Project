import socket

SERVER_HOST = '127.0.0.1'  # 服务器的主机名或IP地址
SERVER_PORT = 12345        # 服务器正在监听的端口

with open(r'input\face_recognition_result.txt', 'r', encoding='utf-8') as file:
    content = file.read()

words = content.split()

total_words = len(words)
count_zzb = words.count('zzb')

proportion_zzb = (count_zzb / total_words) * 100

# 打印结果
print(f'人物出现在镜头前的概率: {proportion_zzb:.2f}%')

message = f"{proportion_zzb:.2f}"

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # 连接到服务器
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"已连接到 {SERVER_HOST}:{SERVER_PORT}")

    # 发送数据
    client_socket.sendall(message.encode('utf-8'))
    print(f"已发送数据: {message}")

    # 接收服务器响应（可选）
    response = client_socket.recv(1024)
    print(f"服务器响应: {response.decode('utf-8')}")

finally:
    # 关闭socket连接
    client_socket.close()
    print("连接已关闭")

