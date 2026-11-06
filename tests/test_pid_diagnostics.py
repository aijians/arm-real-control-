"""Read-only serial diagnostics against an IPC motor, never a device node."""

import importlib.util
from pathlib import Path
import unittest

from control.protocol import make_frame
from pid_tuner.protocol import encode_pid


# Share the existing socket transport fixture while also supporting direct
# execution of this file and unittest discovery from the repository root.
_fixture_spec = importlib.util.spec_from_file_location(
    "_pid_diagnostic_transport_helpers",
    Path(__file__).with_name("test_pid_transport.py"),
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
motor_emulator = _fixture_module.motor_emulator
send_reply = _fixture_module.send_reply


def product_info():
    return (
        b"DRIVER-TEST".ljust(20, b"\0")
        + b"MOTOR-TEST".ljust(20, b"\0")
        + b"MOTOR-ID-17".ljust(12, b"\0")
        + (20).to_bytes(2, "little")
        + (30).to_bytes(2, "little")
        + (235).to_bytes(2, "little")
    )


def valid_payload(request):
    if request.command == 0x94:
        return (15678).to_bytes(4, "little")
    if request.command == 0x9C:
        return bytes.fromhex("20 00 00 00 00 00 00")
    if request.command == 0x9A:
        return bytes.fromhex("20 60 09 00 00 00 00")
    if request.command == 0x12:
        return product_info()
    if request.command == 0x40:
        return request.payload[:1] + encode_pid(300, 30, 7)
    raise AssertionError(f"Unexpected diagnostic command: 0x{request.command:02X}")


class DiagnosticWireEvidenceTests(unittest.TestCase):
    def test_local_echo_is_skipped_before_the_actual_parameter_reply(self):
        raw = encode_pid(300, 30, 7)

        def respond(request, peer):
            echo = make_frame(request.command, request.motor_id, request.payload)
            reply = make_frame(0x40, request.motor_id, request.payload[:1] + raw)
            send_reply(peer, echo + reply)

        with motor_emulator(respond) as (device, port, received, logs):
            self.assertEqual(device.read_parameter(0x96), raw)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].command, 0x40)

    def test_echo_without_a_real_reply_is_a_timeout_with_received_evidence(self):
        def respond(request, peer):
            send_reply(peer, make_frame(request.command, request.motor_id, request.payload))

        with motor_emulator(respond, timeout=0.025) as (device, port, received, logs):
            with self.assertRaises(TimeoutError) as caught:
                device.read_parameter(0x96)
            details = caught.exception.details
            packet = make_frame(0x40, 17, b"\x96\0")
            self.assertEqual(bytes.fromhex(details["tx_hex"]), packet)
            self.assertEqual(details["received_bytes"], len(packet))
            self.assertEqual(bytes.fromhex(details["raw_rx_hex"]), packet)
            self.assertTrue(details["unmatched_frames"])
            self.assertEqual(bytes.fromhex(details["incomplete_hex"]), b"")
            self.assertEqual(len(received), 1, "This transaction should send one read request")

    def test_bad_checksum_bytes_are_preserved_even_without_a_parsed_frame(self):
        bad = bytearray(make_frame(0x40, 17, b"\x96" + encode_pid(300, 30, 7)))
        bad[-1] ^= 1

        def respond(request, peer):
            send_reply(peer, bytes(bad))

        with motor_emulator(respond, timeout=0.025) as (device, port, received, logs):
            with self.assertRaises(TimeoutError) as caught:
                device.read_parameter(0x96)
            details = caught.exception.details
            self.assertEqual(details["received_bytes"], len(bad))
            self.assertEqual(bytes.fromhex(details["raw_rx_hex"]), bytes(bad))
            self.assertTrue(any("RX_BYTES" in message for level, message in logs))

    def test_wrong_motor_reply_is_reported_as_unmatched(self):
        wrong_motor = make_frame(0x40, 18, b"\x96" + encode_pid(300, 30, 7))

        def respond(request, peer):
            send_reply(peer, wrong_motor)

        with motor_emulator(respond, timeout=0.025) as (device, port, received, logs):
            with self.assertRaises(TimeoutError) as caught:
                device.read_parameter(0x96)
            details = caught.exception.details
            self.assertEqual(details["received_bytes"], len(wrong_motor))
            self.assertEqual(bytes.fromhex(details["raw_rx_hex"]), wrong_motor)
            self.assertTrue(details["unmatched_frames"])

    def test_partial_reply_remains_available_in_timeout_evidence(self):
        partial = make_frame(0x40, 17, b"\x96" + encode_pid(300, 30, 7))[:8]

        def respond(request, peer):
            send_reply(peer, partial)

        with motor_emulator(respond, timeout=0.025) as (device, port, received, logs):
            with self.assertRaises(TimeoutError) as caught:
                device.read_parameter(0x96)
            details = caught.exception.details
            self.assertEqual(details["received_bytes"], len(partial))
            self.assertEqual(bytes.fromhex(details["raw_rx_hex"]), partial)
            self.assertEqual(bytes.fromhex(details["incomplete_hex"]), partial)


class ReadOnlyProbeTests(unittest.TestCase):
    def assert_probe_commands(self, received):
        commands = [frame.command for frame in received]
        self.assertEqual(commands, [0x94, 0x9C, 0x9A, 0x12, 0x40, 0x40])
        self.assertEqual([frame.payload for frame in received[-2:]], [b"\x96\0", b"\x97\0"])
        self.assertTrue(all(frame.motor_id == 17 for frame in received))
        self.assertTrue(all(command in {0x94, 0x9C, 0x9A, 0x12, 0x40} for command in commands))
        self.assertTrue({0x30, 0x42, 0x44, 0x81, 0x88, 0xA6}.isdisjoint(commands))

    def test_successful_probe_decodes_the_full_58_byte_product_reply(self):
        self.assertEqual(len(product_info()), 58)

        def respond(request, peer):
            echo = make_frame(request.command, request.motor_id, request.payload)
            reply = make_frame(request.command, request.motor_id, valid_payload(request))
            send_reply(peer, echo + reply)

        with motor_emulator(respond) as (device, port, received, logs):
            report = device.diagnose()
            self.assert_probe_commands(received)
            steps = report["steps"]
            self.assertEqual(len(steps), 6)
            self.assertTrue(all(step["ok"] for step in steps))
            self.assertEqual([step["command"] for step in steps],
                             ["0x94", "0x9C", "0x9A", "0x12", "0x40", "0x40"])
            self.assertTrue(all(step["name"] for step in steps))
            product = steps[3]["decoded"]
            self.assertIsInstance(product, dict)
            self.assertIn("DRIVER-TEST", str(product))
            self.assertIn("MOTOR-TEST", str(product))
            self.assertIsInstance(report["summary"], list)
            self.assertTrue(all(isinstance(item, str) for item in report["summary"]))

    def test_failed_steps_are_recorded_and_later_read_only_probes_continue(self):
        def respond(request, peer):
            if request.command == 0x94:
                return
            if request.command == 0x9A:
                bad = bytearray(make_frame(0x9A, request.motor_id, valid_payload(request)))
                bad[-1] ^= 1
                send_reply(peer, bytes(bad))
                return
            if request.command == 0x40 and request.payload[0] == 0x96:
                send_reply(peer, make_frame(0x40, request.motor_id, b"\x97" + encode_pid(300, 30, 7)))
                return
            send_reply(peer, make_frame(request.command, request.motor_id, valid_payload(request)))

        with motor_emulator(respond, timeout=0.025) as (device, port, received, logs):
            report = device.diagnose()
            self.assert_probe_commands(received)
            steps = report["steps"]
            self.assertEqual([step["ok"] for step in steps], [False, True, False, True, False, True])
            for index in (0, 2, 4):
                self.assertTrue(steps[index]["error"])
                self.assertIsInstance(steps[index]["details"], dict)
            self.assertEqual(steps[0]["details"]["received_bytes"], 0)
            self.assertGreater(steps[2]["details"]["received_bytes"], 0)
            self.assertTrue(steps[4]["details"]["unmatched_frames"])
            self.assertTrue(report["summary"])

    def test_silent_motor_returns_all_failed_steps_without_unsafe_commands(self):
        with motor_emulator(lambda *_: None, timeout=0.02) as (device, port, received, logs):
            report = device.diagnose()
            self.assert_probe_commands(received)
            self.assertEqual(len(report["steps"]), 6)
            self.assertTrue(all(not step["ok"] for step in report["steps"]))
            self.assertTrue(all(step["details"]["received_bytes"] == 0 for step in report["steps"]))
            self.assertTrue(report["summary"])


if __name__ == "__main__":
    unittest.main()
