import socket
import face_input 
from face_input import face_input # 确保这里正确导入了 face_input 类

class ClientSocket:
    def __init__(self, host, port):
        self.SERVER_HOST = host
        self.SERVER_PORT = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_server(self):
        try:
            self.client_socket.connect((self.SERVER_HOST, self.SERVER_PORT))
            print(f"已连接到 {self.SERVER_HOST}:{self.SERVER_PORT}")
        except Exception as e:
            print(f"连接失败: {e}")

    def send_data(self, message):
        try:
            self.client_socket.sendall(message.encode('utf-8'))
            print(f"已发送数据: {message}")
        except Exception as e:
            print(f"发送数据失败: {e}")

    def receive_response(self):
        try:
            response = self.client_socket.recv(1024)
            print(f"服务器响应: {response.decode('utf-8')}")
        except Exception as e:
            print(f"接收响应失败: {e}")

    def close_connection(self):
        self.client_socket.close()
        print("连接已关闭")

# 使用类
if __name__ == "__main__":
    client = ClientSocket('127.0.0.1', 12345)
    client.connect_to_server()
    
    # 创建 face_input 类的实例
    face_recognition = face_input()
    print("1")
    # 启动人脸识别过程
    face_recognition.get_frame() # 理论上这个语句要写进循环？
    print("2")
    # 获取出现频率最高的名字及其比例
    most_common_name, proportion = face_recognition.proportion()
    #print(face_recognition.proportion())
    print("3")
    # 格式化返回值
    result_message = f"姓名: {most_common_name}, 概率: {proportion:.2%}"
    print(result_message)
    # 发送面部识别结果
    client.send_data(result_message)
    client.receive_response()
    
    client.close_connection()