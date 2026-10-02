"""Optional ROS 2 Humble bridge on CMU's topic names (rclpy imported lazily).

  /autohand_node/state_autohand  JointState out: 12 fingers [0,1] + roll (deg)
  /autohand_node/cmd_autohand    JointState in:  same layout
  /dexkit/gantry/state           JointState out: x y z (mm)
  /dexkit/gantry/cmd             JointState in:  x y z (mm)

    source /opt/ros/humble/setup.bash && dexkit-ros-bridge [--mock --yes]
"""

from __future__ import annotations

import numpy as np


def main(argv: list[str] | None = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as e:
        raise SystemExit(f"ROS 2 not available ({e}); source /opt/ros/humble/setup.bash first") from e

    from dexkit.cli import add_gantry_flags, common_parser, init, open_session
    from dexkit.hw.base import N_FINGERS

    p = common_parser(__doc__ or "")
    add_gantry_flags(p)
    p.add_argument("--rate", type=float, default=15.0, help="state publish rate (Hz)")
    args = p.parse_args(argv)
    init(args)

    with open_session(args, need_hand=True, want_gantry=True) as s:
        assert s.hand is not None and s.hand_cfg is not None
        names = s.hand_cfg.names + ["roll"]

        class Bridge(Node):
            def __init__(self) -> None:
                super().__init__("dexkit_bridge")
                self.hand_pub = self.create_publisher(JointState, "/autohand_node/state_autohand", 100)
                self.create_subscription(JointState, "/autohand_node/cmd_autohand", self.on_hand_cmd, 100)
                self.gantry_pub = self.create_publisher(JointState, "/dexkit/gantry/state", 10)
                self.create_subscription(JointState, "/dexkit/gantry/cmd", self.on_gantry_cmd, 10)
                self.create_timer(1.0 / args.rate, self.publish_state)

            def on_hand_cmd(self, msg: JointState) -> None:
                pos = np.asarray(msg.position, dtype=float)
                if len(pos) < N_FINGERS:
                    self.get_logger().warn(f"cmd needs >= {N_FINGERS} positions, got {len(pos)}")
                    return
                roll = float(pos[N_FINGERS]) if len(pos) > N_FINGERS else s.hand.last_command[1]
                s.estop.check()
                s.hand.set_targets(np.clip(pos[:N_FINGERS], 0, 1), roll)

            def on_gantry_cmd(self, msg: JointState) -> None:
                if s.gantry is None or len(msg.position) < 3:
                    return
                s.estop.check()
                s.gantry.move_to(*[float(v) for v in msg.position[:3]], wait=False)

            def publish_state(self) -> None:
                hs = s.hand.get_state()
                m = JointState()
                m.header.stamp = self.get_clock().now().to_msg()
                m.name = names
                m.position = [float(v) for v in np.clip(hs.fingers, 0, 1)] + [float(hs.roll_deg)]
                self.hand_pub.publish(m)
                if s.gantry is not None:
                    gs = s.gantry.get_state()
                    g = JointState()
                    g.header.stamp = m.header.stamp
                    g.name = ["x", "y", "z"]
                    g.position = [float(v) for v in gs.xyz]
                    self.gantry_pub.publish(g)

        rclpy.init()
        node = Bridge()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
