from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

def main():
    # 创建一个授权对象
    authorizer = DummyAuthorizer()

    # 添加用户，设置用户名、密码、目录和权限
    authorizer.add_user('user', '12345', 'C:\\Users\\86138\\Pictures\\Camera Roll',perm='elradfmwMT')

    # 添加匿名用户，设置目录和权限
    authorizer.add_anonymous('C:\\Users\\86138\\Pictures\\Camera Roll')

    # 创建一个 FTP 处理程序并设置授权对象
    handler = FTPHandler
    handler.authorizer = authorizer

    # 设置允许访问的 IP 地址列表
    allowed_ips = ['127.0.0.1', '192.168.1.1']

    # 定义一个函数来检查客户端 IP 是否在允许列表中
    def check_client_ip(self):
        
        if(self.remote_ip in allowed_ips):
            return True
        else:
            self.close_when_done()
    # 将自定义的 IP 检查函数设置到处理程序中
    handler.on_connect = check_client_ip

    # 启动 FTP 服务器
    server = FTPServer(('0.0.0.0', 21), handler)
    server.serve_forever()

if __name__ == '__main__':
    main()