"""
Arduino UART Driver for Raspberry Pi
Handles serial communication with Arduino Uno
Camera-only version - No IR sensors
"""

import serial
import json
import threading
import time
import logging
from typing import Optional, Dict, Callable
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ArduinoDriver:
    """
    Arduino UART Communication Driver
    Camera-only version - handles motors only
    """
    
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        """
        Initialize Arduino driver
        
        Args:
            port: Serial port path (e.g., /dev/ttyACM0, /dev/ttyUSB0)
            baudrate: Communication baudrate (default: 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        
        # Current state
        self.connected = False
        self.left_speed = 0
        self.right_speed = 0
        
        # Sensor data (simplified - no sensors)
        self.sensor_data = {
            'left_speed': 0,
            'right_speed': 0,
            'uptime': 0,
            'mode': 'camera_only'
        }
        
        # Callbacks
        self.sensor_callback: Optional[Callable] = None
        
        # ✅ FIX RACE CONDITION: Response queue cho wait_response=True
        # _read_loop sẽ push responses vào queue này
        # send_command() sẽ pop từ queue thay vì poll serial trực tiếp
        self.response_queue: Queue = Queue(maxsize=10)
        
        # Thread control
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        
        # ✅ SAFETY: Heartbeat thread để giữ Arduino watchdog alive
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_interval = 1.0  # Send PING every 1 second (< 2s timeout)
        
        # Try to connect
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to Arduino via UART
        
        Returns:
            True if connected successfully
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            
            # Wait for Arduino to boot
            time.sleep(5)
            
            # Flush buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Send ping to check connection
            response = self.send_command({'cmd': 'PING'})
            if response and response.get('status') == 'ok':
                self.connected = True
                logger.info(f"Connected to Arduino on {self.port} (Camera-only mode)")
                
                # Start reading thread
                self.running = True
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self.read_thread.start()
                
                # ✅ SAFETY: Start heartbeat thread để giữ Arduino watchdog alive
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self.heartbeat_thread.start()
                
                return True
            else:
                logger.warning("Arduino did not respond to PING")
                return False
                
        except serial.SerialException as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Arduino: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
        
        if self.serial and self.serial.is_open:
            self.serial.close()
        
        self.connected = False
        logger.info("Disconnected from Arduino")
    
    def send_command(self, command: Dict, wait_response: bool = True) -> Optional[Dict]:
        """
        Send JSON command to Arduino
        
        ✅ FIX RACE CONDITION: Dùng response queue thay vì poll serial trực tiếp
        
        Args:
            command: Command dictionary
            wait_response: Wait for acknowledgment
            
        Returns:
            Response dictionary or None
        """
        if not self.serial:
            logger.warning("Cannot send command: Serial port not open")
            return None
        
        try:
            # Serialize to JSON
            json_str = json.dumps(command) + '\n'
            
            # Send via serial
            self.serial.write(json_str.encode('utf-8'))
            self.serial.flush()
            
            logger.debug(f"Sent: {command}")
            
            # Wait for response if requested
            if wait_response:
                # ============================================================
                # ✅ FIX: Đọc từ response_queue thay vì poll serial
                # ============================================================
                # _read_loop sẽ push 'status' responses vào queue
                # Không có race condition vì chỉ _read_loop đọc serial
                # ============================================================
                try:
                    # Wait up to 3 seconds for response
                    response = self.response_queue.get(timeout=3.0)
                    logger.debug(f"Received ACK: {response}")
                    return response
                except Empty:
                    logger.warning(f"Command response timeout: {command}")
                    return None
            
            return {'status': 'sent'}
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return None
    
    def set_motors(self, left_speed: int, right_speed: int):
        """
        Set motor speeds
        
        Args:
            left_speed: -255 to 255 (negative = backward)
            right_speed: -255 to 255 (negative = backward)
        """
        left_speed = max(-255, min(255, left_speed))
        right_speed = max(-255, min(255, right_speed))
        
        self.left_speed = left_speed
        self.right_speed = right_speed
        
        command = {
            'cmd': 'MOVE',
            'left': left_speed,
            'right': right_speed
        }
        
        self.send_command(command, wait_response=False)
    
    def forward(self, speed: int = 200):
        """Move forward"""
        self.set_motors(speed, speed)
    
    def backward(self, speed: int = 200):
        """Move backward"""
        self.set_motors(-speed, -speed)
    
    def turn_left(self, speed: int = 150):
        """Turn left (rotate in place)"""
        self.set_motors(-speed, speed)
    
    def turn_right(self, speed: int = 150):
        """Turn right (rotate in place)"""
        self.set_motors(speed, -speed)
    
    def stop(self):
        """Stop motors"""
        self.set_motors(0, 0)
        command = {'cmd': 'STOP'}
        self.send_command(command, wait_response=False)
    
    def get_speeds(self):
        """Get current motor speeds"""
        return (self.left_speed, self.right_speed)
    
    def get_sensor_data(self) -> Dict:
        """Get latest sensor data"""
        return self.sensor_data.copy()
    
    def set_sensor_callback(self, callback: Callable):
        """
        Set callback function for sensor data updates
        
        Args:
            callback: Function to call when new sensor data arrives
        """
        self.sensor_callback = callback
    
    def _read_loop(self):
        """
        Background thread to read sensor data from Arduino
        
        ✅ FIX RACE CONDITION: Push 'status' responses vào queue cho send_command()
        """
        logger.info("Arduino read loop started (camera-only mode)")
        
        while self.running:
            try:
                if self.serial and self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    
                    if line:
                        try:
                            data = json.loads(line)
                            
                            # ============================================================
                            # ✅ FIX: Route messages dựa trên type
                            # ============================================================
                            
                            # 1. Status responses (ACK/PONG) → response_queue
                            if 'status' in data:
                                if data['status'] == 'ok':
                                    # Push vào queue cho send_command() đang chờ
                                    try:
                                        self.response_queue.put_nowait(data)
                                    except:
                                        # Queue full, drop oldest
                                        try:
                                            self.response_queue.get_nowait()
                                            self.response_queue.put_nowait(data)
                                        except:
                                            pass
                                
                                elif data['status'] == 'ready':
                                    logger.info(f"Arduino ready (camera-only mode): {data}")
                                
                                elif data['status'] == 'error':
                                    logger.error(f"Arduino error: {data.get('message', 'Unknown')}")
                            
                            # 2. Motor status data → sensor_callback
                            elif 'left_speed' in data or 'right_speed' in data:
                                self.sensor_data.update(data)
                                
                                # Call callback if set
                                if self.sensor_callback:
                                    self.sensor_callback(data)
                        
                        except json.JSONDecodeError:
                            logger.debug(f"Non-JSON line: {line}")
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                time.sleep(0.1)
        
        logger.info("Arduino read loop stopped")
    
    def _heartbeat_loop(self):
        """
        Background thread to send periodic PING to Arduino
        
        ✅ SAFETY: Keeps Arduino watchdog alive
        - Arduino has 2-second timeout
        - We send PING every 1 second (safety margin)
        - If this thread stops (Pi crash), Arduino auto-stops motors
        """
        logger.info("Arduino heartbeat loop started")
        
        while self.running:
            try:
                # Send PING to keep Arduino watchdog alive
                # Don't wait for response (non-blocking)
                self.send_command({'cmd': 'PING'}, wait_response=False)
                
                # Sleep for heartbeat interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(1.0)
        
        logger.info("Arduino heartbeat loop stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.disconnect()
        logger.info("Arduino driver cleaned up")
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass