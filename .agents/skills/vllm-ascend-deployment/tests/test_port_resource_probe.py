from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import port_resource_probe  # noqa: E402


class PortResourceProbeTests(unittest.TestCase):
    def test_provider_order_is_kept_without_external_preference(self) -> None:
        checks: list[int] = []

        def checker(port: int) -> dict[str, object]:
            checks.append(port)
            return {"port": port, "status": "available"}

        result = port_resource_probe.select_provider_ports(
            ["8120", "8117", "8121"],
            required_count=2,
            checker=checker,
        )
        self.assertEqual(checks, [8120, 8117])
        self.assertEqual(result["selected"], [8120, 8117])
        self.assertIsNone(result["preferred_port"])

    def test_external_preference_is_applied_only_to_provider_candidates(self) -> None:
        result = port_resource_probe.select_provider_ports(
            [8120, 8117],
            required_count=1,
            preferred=8117,
            checker=lambda port: {"port": port, "status": "available"},
        )
        self.assertEqual(result["selected"], [8117])
        with self.assertRaises(port_resource_probe.PortProbeError) as context:
            port_resource_probe.ordered_provider_ports([8120], preferred=8117)
        self.assertEqual(
            context.exception.failure_type,
            "preferred_port_not_provider_candidate",
        )

    def test_occupied_port_is_not_selected(self) -> None:
        def checker(port: int) -> dict[str, object]:
            return {
                "port": port,
                "status": "unavailable",
                "failure_type": "port_unavailable",
                "message": "already bound",
            }

        with self.assertRaises(port_resource_probe.PortProbeError) as context:
            port_resource_probe.select_provider_ports([8120], checker=checker)
        self.assertEqual(context.exception.failure_type, "port_unavailable")


if __name__ == "__main__":
    unittest.main()
