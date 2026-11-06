"""In-process motor emulator; these tests never open a device node."""

from contextlib import contextmanager
import os
import select
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from control.motors import _Port
from control.protocol import extract_frame, make_frame, parse_frame
from pid_tuner.device import DemoDevice, SerialDevice
from pid_tuner.protocol import encode_pid
from pid_tuner.service import Tuner


CONNECTION = {"device": "/virtual/upper-only", "baudrate": 115200,
              "motor_id": 17, "sign": 1, "ratio": 1.0, "offset_deg": 0.0}


def send_reply(peer, data):
    # Local IPC uses the same read/write syscalls as _Port, without any
    # network calls; some workspace sandboxes deny socket.send/sendall.
    os.write(peer.fileno(), data)


class SocketPort(_Port):
    """Use the real _Port.send pacing on a socket instead of a serial device."""
    def __init__(self, client):
        super().__init__(CONNECTION["device"], CONNECTION["baudrate"])
        self.fd = os.dup(client.fileno())
        self.send_entered = threading.Event()
        self.sent_at = []

    def send(self, *args, **kwargs):
        self.send_entered.set()
        super().send(*args, **kwargs)
        self.sent_at.append(time.monotonic())


@contextmanager
def motor_emulator(respond, timeout=0.12):
    client, peer = socket.socketpair()
    client.setblocking(False)
    peer.setblocking(False)
    port = SocketPort(client)
    client.close()
    received, errors = [], []
    quit_event = threading.Event()
    logs = []

    def run():
        buffer = bytearray()
        try:
            while not quit_event.is_set():
                if not select.select([peer], [], [], 0.02)[0]:
                    continue
                data = peer.recv(4096)
                if not data:
                    return
                buffer.extend(data)
                while True:
                    frame = extract_frame(buffer)
                    if frame is None:
                        break
                    received.append(frame)
                    respond(frame, peer)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except BaseException as exc:
            errors.append(exc)

    with patch("pid_tuner.device._Port", return_value=port):
        device = SerialDevice(dict(CONNECTION), lambda *args: logs.append(args), timeout=timeout)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield device, port, received, logs
    finally:
        quit_event.set()
        device.close()
        worker.join(0.3)
        peer.close()
        if errors:
            raise errors[0]


class SerialTransportTests(unittest.TestCase):
    def test_fragmented_reply_ignores_noise_bad_checksums_and_wrong_transaction(self):
        raw = encode_pid(300, 30, 7)

        def respond(request, peer):
            correct = make_frame(0x40, 17, b"\x97" + raw)
            corrupt = bytearray(correct)
            corrupt[-1] ^= 1
            send_reply(peer, b"noise" + make_frame(0x40, 18, b"\x97" + bytes(6))
                       + make_frame(0x42, 17, b"\x97\0")
                       + make_frame(0x40, 17, b"\x96" + bytes(6)) + corrupt)
            for piece in (correct[:3], correct[3:8], correct[8:]):
                send_reply(peer, piece)
                time.sleep(0.002)

        with motor_emulator(respond) as (device, port, received, logs):
            self.assertEqual(device.read_parameter(0x97), raw)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].payload, b"\x97\0")

    def test_failed_write_ack_is_not_retried(self):
        def respond(request, peer):
            send_reply(peer, make_frame(request.command, 17, b"\x97\x01"))

        with motor_emulator(respond) as (device, port, received, logs):
            with self.assertRaisesRegex(ValueError, "rejected"):
                device.write_parameter(0x97, encode_pid(300, 30, 0))
            self.assertEqual(len(received), 1)

    def test_write_timeout_is_not_retried(self):
        with motor_emulator(lambda *_: None, timeout=0.04) as (device, port, received, logs):
            with self.assertRaises(TimeoutError):
                device.write_parameter(0x97, encode_pid(300, 30, 0), persistent=True)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].command, 0x44)

    def test_wrong_parameter_ack_cannot_confirm_a_write(self):
        def respond(request, peer):
            send_reply(peer, make_frame(request.command, 17, b"\x96\0"))

        with motor_emulator(respond, timeout=0.04) as (device, port, received, logs):
            with self.assertRaises(TimeoutError):
                device.write_parameter(0x97, encode_pid(300, 30, 0))
            self.assertEqual(len(received), 1)

    def test_real_port_reply_gap_is_retained(self):
        def respond(request, peer):
            send_reply(peer, make_frame(request.command, 17, request.payload[:1] + bytes(6)))

        with motor_emulator(respond) as (device, port, received, logs):
            device.read_parameter(0x96)
            device.read_parameter(0x97)
            self.assertGreaterEqual(port.sent_at[1]-port.sent_at[0], _Port.REPLY_GAP_S-0.0005)
            self.assertTrue(all(frame.motor_id == 17 for frame in received))

    def test_telemetry_preserves_signed_wire_fields_and_axis_mapping(self):
        replies = {0x94: (14000).to_bytes(4, "little"),
                   0x9C: bytes.fromhex("F6 FF FF F6 FF FF FF"),
                   0x9A: bytes.fromhex("F6 60 09 83 FF 10 40")}

        def respond(request, peer):
            send_reply(peer, make_frame(request.command, 17, replies[request.command]))

        with motor_emulator(respond) as (device, port, received, logs):
            device.connection.update(sign=-1, ratio=2., offset_deg=180.)
            row = device.telemetry()
            self.assertEqual(row["angle_deg"], 20.)
            self.assertEqual(row["motor_angle_deg"], 140.)
            self.assertEqual(row["speed_deg_s"], 5.)
            self.assertEqual(row["temperature_c"], -10)
            self.assertEqual(row["bus_current_a"], -1.25)
            self.assertEqual(row["iq_or_power_raw"], -1)
            self.assertEqual(row["voltage_v"], 24.)
            self.assertEqual(row["encoder_raw"], 65535)
            self.assertEqual(row["error_flags"], ["stall"])
            self.assertEqual([frame.command for frame in received], [0x94, 0x9C, 0x9A])

    def test_move_maps_only_the_configured_axis_to_a6(self):
        def respond(request, peer):
            send_reply(peer, make_frame(0xA6, 17, bytes(7)))

        with motor_emulator(respond) as (device, port, received, logs):
            device.connection.update(sign=-1, ratio=2., offset_deg=180.)
            device.move(20., 2., 145.)
            self.assertEqual(len(received), 1)
            request = received[0]
            self.assertEqual((request.command, request.motor_id), (0xA6, 17))
            self.assertEqual(request.payload, bytes.fromhex("01 B0 36 00 90 01 00 00"))

    def test_pending_read_yields_to_stop(self):
        requested = threading.Event()

        def respond(request, peer):
            if request.command == 0x40:
                requested.set()
            elif request.command == 0x81:
                send_reply(peer, make_frame(0x81, 17))

        with motor_emulator(respond, timeout=1.) as (device, port, received, logs):
            result = []

            def read():
                try:
                    device.read_parameter(0x96)
                except BaseException as exc:
                    result.append(exc)

            reader = threading.Thread(target=read)
            reader.start()
            self.assertTrue(requested.wait(0.2))
            started = time.monotonic()
            device.interrupt()
            device.stop()
            reader.join(0.2)
            self.assertFalse(reader.is_alive())
            self.assertLess(time.monotonic()-started, 0.2)
            self.assertIsInstance(result[0], InterruptedError)
            self.assertEqual([frame.command for frame in received], [0x40, 0x81])

    def test_stop_during_send_gap_prevents_the_original_write(self):
        def respond(request, peer):
            send_reply(peer, make_frame(request.command, 17, b"\x97\0"))

        with motor_emulator(respond) as (device, port, received, logs):
            # Force the 5 ms pacing window open long enough to deterministically
            # inject the stop after _transaction's pre-send interruption check.
            port._next_send_at = time.monotonic()+0.06
            result = []

            def write():
                try:
                    device.write_parameter(0x97, encode_pid(300, 30, 0))
                except BaseException as exc:
                    result.append(exc)

            writer = threading.Thread(target=write)
            writer.start()
            self.assertTrue(port.send_entered.wait(0.2))
            device.interrupt()
            writer.join(0.3)
            self.assertFalse(writer.is_alive())
            self.assertTrue(result, "A stopped write must fail before transmission")
            self.assertEqual(received, [], "No 0x42 packet may be sent after stop during pacing")


class ConfiguredAxisAndDemoTests(unittest.TestCase):
    def test_tuner_constructs_only_the_configured_upper_axis(self):
        created = []

        def factory(connection, logger):
            created.append(dict(connection))
            return DemoDevice(connection, logger)

        config = {"hardware": {"baudrate": 115200, "max_speed_deg_s": 18., "motors": [
            {"name": "upper", "device": "/virtual/upper-only", "id": 17,
             "sign": 1, "ratio": 1., "offset_deg": 0., "limits_deg": [0., 180.]},
            {"name": "lower", "device": "/must-never-open/lower", "id": 2,
             "sign": 1, "ratio": 1., "offset_deg": 0., "limits_deg": [0., 90.]},
        ]}}
        with tempfile.TemporaryDirectory() as runtime:
            tuner = Tuner(config, {"runtime_dir": runtime}, device_factory=factory, start_worker=False)
            try:
                tuner.connect()
                self.assertEqual(len(created), 1)
                self.assertEqual(created[0]["device"], "/virtual/upper-only")
                self.assertEqual(created[0]["motor_id"], 17)
            finally:
                tuner.close()

    def test_demo_uses_the_same_axis_mapping_and_a6_format(self):
        logs = []
        connection = {**CONNECTION, "sign": -1, "ratio": 1.5, "offset_deg": 350.}
        device = DemoDevice(connection, lambda *args: logs.append(args))
        row = device.telemetry()
        expected_motor = 350.-1.5*row["angle_deg"]
        self.assertAlmostEqual(row["motor_angle_deg"], expected_motor)
        target = row["angle_deg"]-1
        device.move(target, 2., row["motor_angle_deg"])
        packet = bytes.fromhex(logs[-2][1][3:])
        frame = parse_frame(packet)
        self.assertEqual(frame.command, 0xA6)
        self.assertEqual(frame.motor_id, 17)
        self.assertEqual(frame.payload[0], 0)
        self.assertEqual(int.from_bytes(frame.payload[1:4], "little"), round((350.-1.5*target)*100))
        self.assertEqual(int.from_bytes(frame.payload[4:8], "little"), 300)


if __name__ == "__main__":
    unittest.main()
