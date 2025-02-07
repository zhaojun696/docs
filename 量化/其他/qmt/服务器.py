import json
import socket
import http.server
from urllib.parse import urlparse, parse_qs



class logic:
    @staticmethod
    def index(query_params):
        return {
            'iopv':get_etf_iopv("510050.SH")
        }
    

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query) 

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        router=path[1:]
        if(hasattr(logic,router)):
            response=getattr(logic,router)
            try:
                response=response(query_params)
                response={'code':0,'data':response}
            except Exception as err:
                print('发生错误:',err)
                response={'code':1,'data':str(err)}
            
        else:
            response={'code':404,'data':None}

        response = json.dumps(response)
        self.wfile.write(response.encode('utf-8'))

def check_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def run(server_class=http.server.HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    
    if check_port_in_use(port):
        raise RuntimeError(f'Port {port} is already in use.')

    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

run(port=8085)