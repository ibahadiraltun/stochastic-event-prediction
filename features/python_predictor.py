#!/usr/bin/env python3
"""
Example Python predictor for STEP Embedding

This script demonstrates how to integrate a neural network (e.g., GNN) with STEP embedding.
It connects to shared memory to receive events and return probability predictions.
"""

import mmap
import struct
import time
import random
import sys
import signal
import os

# Shared memory configuration (must match C++ side)
SHM_NAME = "/step_embedding_shm"
SHM_SIZE = 4096

# Memory layout offsets
REQUEST_FLAG_OFFSET = 0      # 4 bytes: 1 = new request, 0 = read
RESPONSE_FLAG_OFFSET = 4     # 4 bytes: 1 = new response, 0 = read
REQUEST_DATA_OFFSET = 8      # 256 bytes: event string "node_u node_v timestamp"
RESPONSE_DATA_OFFSET = 264   # 8 bytes: probability (double)


class SimplePredictor:
    """Simple shared memory predictor"""

    def __init__(self):
        self.shm_fd = None
        self.shm_map = None
        self.running = True

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

    def stop(self):
        """Stop the predictor"""
        print("\n[Python] Shutting down...")
        self.running = False

    def connect(self):
        """Connect to shared memory"""

        # Wait for C++ to create shared memory
        max_attempts = 60  # Wait up to 30 seconds
        for attempt in range(max_attempts):
            try:
                # Try multiple methods to open shared memory
                # Method 1: Direct os.open (Linux style)
                try:
                    self.shm_fd = os.open(SHM_NAME, os.O_RDWR)
                    self.shm_map = mmap.mmap(self.shm_fd, SHM_SIZE)
                    print(f"[Python] Connected to shared memory: {SHM_NAME}")
                    return True
                except:
                    pass

                # Method 2: Try with full path (macOS sometimes needs this)
                try:
                    if sys.platform == "darwin":
                        # On macOS, shared memory might be in /private/tmp
                        import fcntl
                        # Use shm_open directly via ctypes
                        import ctypes
                        libc = ctypes.CDLL("libc.dylib")
                        shm_open_func = libc.shm_open
                        shm_open_func.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
                        shm_open_func.restype = ctypes.c_int
                        O_RDWR = 2
                        fd = shm_open_func(SHM_NAME.encode(), O_RDWR, 0o666)
                        if fd >= 0:
                            self.shm_fd = fd
                            self.shm_map = mmap.mmap(fd, SHM_SIZE)
                            print(f"[Python] Connected to shared memory: {SHM_NAME}")
                            return True
                except Exception as e2:
                    pass

                if attempt == 0:
                    print("[Python] Waiting for C++ to create shared memory...")
                time.sleep(0.5)

            except Exception as e:
                if attempt == 0:
                    print("[Python] Waiting for C++ to create shared memory...")
                time.sleep(0.5)

        print("[Python] Failed to connect - make sure C++ process is running")
        return False

    def read_flag(self, offset):
        """Read a 4-byte integer flag"""
        self.shm_map.seek(offset)
        return struct.unpack('i', self.shm_map.read(4))[0]

    def write_flag(self, offset, value):
        """Write a 4-byte integer flag"""
        self.shm_map.seek(offset)
        self.shm_map.write(struct.pack('i', value))

    def read_request(self):
        """Read event string from shared memory"""
        self.shm_map.seek(REQUEST_DATA_OFFSET)
        data = self.shm_map.read(256)
        # Remove null terminator
        if b'\x00' in data:
            data = data[:data.index(b'\x00')]
        return data.decode('utf-8').strip()

    def write_response(self, probability):
        """Write probability to shared memory"""
        self.shm_map.seek(RESPONSE_DATA_OFFSET)
        self.shm_map.write(struct.pack('d', probability))

    def predict(self, event_info):
        """
        Make prediction for event

        Args:
            event_info: String "node_u node_v timestamp"

        Returns:
            Probability (float)
        """
        # Parse event
        parts = event_info.split()
        node_u = int(parts[0])
        node_v = int(parts[1])
        timestamp = int(parts[2])

        # ========================================
        # YOUR GNN/NEURAL NETWORK PREDICTION HERE
        # ========================================
        # Example:
        #   features = extract_features(node_u, node_v, timestamp)
        #   probability = gnn_model.predict(features)
        #   return probability

        # For demo: return random probability
        probability = random.uniform(0.0, 1.0)

        print(f"[Python] Event ({node_u}, {node_v}) at t={timestamp} -> p={probability:.4f}")
        return probability

    def run(self):
        """Main prediction loop"""
        print("[Python] Starting prediction service...")
        print("[Python] Press Ctrl+C to stop")
        print()

        request_count = 0

        while self.running:
            try:
                # Check for new request
                if self.read_flag(REQUEST_FLAG_OFFSET) == 1:
                    # Read event
                    event_info = self.read_request()

                    # Clear request flag
                    self.write_flag(REQUEST_FLAG_OFFSET, 0)

                    # Make prediction
                    probability = self.predict(event_info)

                    # Write response
                    self.write_response(probability)
                    self.write_flag(RESPONSE_FLAG_OFFSET, 1)

                    request_count += 1
                else:
                    # Small sleep to avoid busy-waiting
                    time.sleep(0.001)

            except Exception as e:
                print(f"[Python] Error: {e}")
                break

        print(f"\n[Python] Processed {request_count} requests")

    def close(self):
        """Close shared memory"""
        if self.shm_map:
            self.shm_map.close()
        if self.shm_fd is not None:
            os.close(self.shm_fd)


def main():
    print("=" * 60)
    print("STEP Embedding - Python GNN Predictor")
    print("=" * 60)
    print()

    predictor = SimplePredictor()

    # Connect to shared memory
    if not predictor.connect():
        return 1

    try:
        # Run prediction loop
        predictor.run()
    finally:
        predictor.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
