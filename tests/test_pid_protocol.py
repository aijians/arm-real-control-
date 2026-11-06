"""Golden wire examples and field validation; no hardware or serial I/O."""

import unittest

from control.protocol import parse_frame
from pid_tuner.protocol import (
    Frame,
    decode_parameter,
    decode_parameter_reply,
    decode_status1,
    decode_status2,
    encode_max_torque_current,
    encode_pid,
    extract_frame,
    position_frame,
    read_parameter_frame,
    validate_write_ack,
    write_parameter_frame,
)


class ParameterProtocolTests(unittest.TestCase):
    def test_golden_parameter_read(self):
        self.assertEqual(read_parameter_frame(1, 0x96),
                         bytes.fromhex("3E 40 01 02 81 96 00 96"))

    def test_golden_ram_write(self):
        self.assertEqual(write_parameter_frame(1, 0x97, encode_pid(300, 30, 0)),
                         bytes.fromhex("3E 42 01 07 88 97 2C 01 1E 00 00 00 E2"))

    def test_golden_rom_write(self):
        self.assertEqual(write_parameter_frame(1, 0x96, encode_pid(50, 0, 0), True),
                         bytes.fromhex("3E 44 01 07 8A 96 32 00 00 00 00 00 C8"))

    def test_successful_motion_examples_stay_identical(self):
        self.assertEqual(position_frame(1, 1, 15896, 6000),
                         bytes.fromhex("3E A6 01 08 ED 01 18 3E 00 70 17 00 00 DE"))
        self.assertEqual(position_frame(2, 0, 5238, 6000),
                         bytes.fromhex("3E A6 02 08 EE 00 76 14 00 70 17 00 00 11"))

    def test_read_reply_preserves_all_six_bytes(self):
        frame = parse_frame(bytes.fromhex("3E 40 01 07 86 97 2C 01 1E 00 00 00 E2"))
        raw = decode_parameter_reply(frame, 0x97)
        self.assertEqual(raw, bytes.fromhex("2C 01 1E 00 00 00"))
        self.assertEqual(decode_parameter(0x97, raw), {"kp": 300, "ki": 30, "kd": 0})

    def test_read_reply_rejects_wrong_command_parameter_or_size(self):
        for frame in (Frame(0x42, 1, b"\x96" + b"\0" * 6),
                      Frame(0x40, 1, b"\x97" + b"\0" * 6),
                      Frame(0x40, 1, b"\x96" + b"\0" * 5)):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                decode_parameter_reply(frame, 0x96)

    def test_uint16_boundaries_and_all_pid_loops(self):
        for param in (0x96, 0x97, 0x98):
            raw = encode_pid(0, 65535, 32768)
            self.assertEqual(decode_parameter(param, raw),
                             {"kp": 0, "ki": 65535, "kd": 32768})

    def test_pid_never_coerces_or_truncates(self):
        for value in (-1, 65536, 50.0, True, False, "50", None):
            for field in range(3):
                values = [1, 2, 3]
                values[field] = value
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    encode_pid(*values)

    def test_raw_parameter_and_ids_are_validated(self):
        for raw in (b"", b"\0" * 5, b"\0" * 7, 6, "000000", [0] * 6):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                write_parameter_frame(1, 0x96, raw)
        for motor_id in (0, 33, True, 1.0):
            with self.subTest(motor_id=motor_id), self.assertRaises(ValueError):
                read_parameter_frame(motor_id, 0x96)
        for param in (0x95, 0x9A, True, 150.0):
            with self.subTest(param=param), self.assertRaises(ValueError):
                read_parameter_frame(1, param)
        with self.assertRaises(ValueError):
            write_parameter_frame(1, 0x96, bytes(6), persistent=1)

    def test_torque_current_is_signed_and_preserves_reserved_bytes(self):
        original = bytes.fromhex("00 80 A1 B2 C3 D4")
        self.assertEqual(decode_parameter(0x99, original),
                         {"value_raw": -32768, "reserved_hex": "a1 b2 c3 d4"})
        modified = encode_max_torque_current(32767, original)
        self.assertEqual(modified, bytes.fromhex("FF 7F A1 B2 C3 D4"))
        frame = parse_frame(write_parameter_frame(1, 0x99, modified))
        self.assertEqual(frame.payload, b"\x99" + modified)
        for value in (-32769, 32768, True, 5.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_max_torque_current(value, original)
        with self.assertRaises(ValueError):
            encode_max_torque_current(1, b"\0\0")

    def test_write_ack_uses_second_byte_for_success(self):
        ack = parse_frame(bytes.fromhex("3E 42 01 02 83 97 00 97"))
        self.assertIsNone(validate_write_ack(ack, 0x42, 0x97))
        self.assertIsNone(validate_write_ack(Frame(0x44, 1, b"\x96\0"), 0x44, 0x96))

    def test_write_ack_rejects_failures_and_mismatches(self):
        for frame in (Frame(0x42, 1, b"\x97\x01"),
                      Frame(0x42, 1, b"\x97\xff"),
                      Frame(0x44, 1, b"\x97\0"),
                      Frame(0x42, 1, b"\x96\0"),
                      Frame(0x42, 1, b"\x97"),
                      Frame(0x42, 1, b"\x97\0\0")):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                validate_write_ack(frame, 0x42, 0x97)
        with self.assertRaises(ValueError):
            validate_write_ack(Frame(0x40, 1, b"\x97\0"), 0x40, 0x97)

    def test_checksum_failure_does_not_deliver_frame(self):
        good = bytes.fromhex("3E 42 01 02 83 97 00 97")
        for index in (4, -1):
            bad = bytearray(good)
            bad[index] ^= 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_frame(bad)
            buffer = bytearray(bad + good)
            self.assertEqual(extract_frame(buffer), Frame(0x42, 1, b"\x97\0"))
            self.assertEqual(buffer, bytearray())

    def test_status1_scaling_signedness_and_flags(self):
        state = decode_status1(bytes.fromhex("FB 60 09 83 FF 10 91"))
        self.assertEqual(state["temperature_c"], -5)
        self.assertEqual(state["voltage_v"], 24.0)
        self.assertEqual(state["bus_current_a"], -1.25)
        self.assertEqual(state["motor_state"], 0x10)
        self.assertEqual(state["error_state"], 0x91)
        self.assertEqual(state["error_flags"], ["low_voltage", "overcurrent", "input_timeout"])

    def test_status2_retains_model_dependent_current_or_power(self):
        self.assertEqual(decode_status2(bytes.fromhex("F6 00 80 C4 FF FF FF")),
                         {"temperature_c": -10, "iq_or_power_raw": -32768,
                          "speed_deg_s": -60, "encoder_raw": 65535})

    def test_status_payload_lengths_are_strict(self):
        for decoder in (decode_status1, decode_status2):
            for raw in (bytes(6), bytes(8), 7, "1234567"):
                with self.subTest(decoder=decoder, raw=raw), self.assertRaises(ValueError):
                    decoder(raw)


if __name__ == "__main__":
    unittest.main()
