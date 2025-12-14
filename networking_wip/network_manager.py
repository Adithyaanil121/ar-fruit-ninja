import socket
import threading
import pickle
import struct

class NetworkManager:
    def __init__(self, is_server, server_ip='localhost', port=5555):
        self.is_server = is_server
        self.server_ip = server_ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.other_players_data = {}
        self.lock = threading.Lock()

    def start_server(self):
        try:
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(1)
            print("Server started, waiting for connection...")
            self.conn, self.addr = self.socket.accept()
            print(f"Connected to: {self.addr}")
            self.connected = True
            threading.Thread(target=self.receive_data, args=(self.conn,), daemon=True).start()
            return True
        except Exception as e:
            print(f"Server error: {e}")
            return False

    def connect_to_server(self):
        try:
            self.socket.connect((self.server_ip, self.port))
            print("Connected to server")
            self.connected = True
            threading.Thread(target=self.receive_data, args=(self.socket,), daemon=True).start()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def send_data(self, data):
        if not self.connected: return
        try:
            # Use pickle to serialize data
            serialized_data = pickle.dumps(data)
            # Prefix with length of data
            message = struct.pack("Q", len(serialized_data)) + serialized_data
            if self.is_server:
                self.conn.sendall(message)
            else:
                self.socket.sendall(message)
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False

    def receive_data(self, sock):
        data = b""
        payload_size = struct.calcsize("Q")
        while self.connected:
            try:
                while len(data) < payload_size:
                    packet = sock.recv(4*1024)
                    if not packet: 
                        self.connected = False
                        return
                    data += packet
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += sock.recv(4*1024)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                received_object = pickle.loads(frame_data)
                
                with self.lock:
                    self.other_players_data = received_object
                    
            except Exception as e:
                print(f"Receive error: {e}")
                self.connected = False
                break

    def get_latest_data(self):
        with self.lock:
            return self.other_players_data
