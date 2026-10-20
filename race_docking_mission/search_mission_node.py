"""Search + dock-approach mission node for the RACE AUV docking redesign.

Phase 1 (search): a staircase of (descend leg, turn leg) pairs, each
descend leg going a little deeper than the last, down to an absolute cap:

  SEARCH_DESCEND: hold x/y/yaw, sink to this leg's target depth (leg 0:
  search_first_descend_limit metres below the start; leg k>=1:
  search_first_descend_limit + k*search_descend_step, capped at
  search_max_depth), watching for an AprilTag detection.
    * Tag detected -> HOLD (freeze the current target, don't move).
    * Reached this leg's target depth with no detection -> SEARCH_TURN.
  SEARCH_TURN: hold x/y/z, rotate yaw a full 360 deg, watching for a
  detection the whole time.
    * Tag detected -> HOLD.
    * Completes 360 deg with no detection, and this wasn't the leg at
      search_max_depth -> next SEARCH_DESCEND leg (deeper).
    * Completes 360 deg with no detection at search_max_depth -> KILL.
  HOLD: entered once tag_debounce_count consecutive detections confirm a
  sighting -- sits here indefinitely (no longer auto-advances):
    * Tag still visible -> pose stays exactly where it was confirmed.
    * Tag lost for hold_lost_timeout seconds -> search for it again
      nearby: a slow triangle-wave wobble in z (+-hold_reacquire_
      z_amplitude at hold_reacquire_z_speed) and yaw (+-hold_reacquire_
      yaw_amplitude_deg at hold_reacquire_yaw_rate_deg_s) around the pose
      that last saw the tag, until it reappears, then snap straight back
      to holding there.
    * start_approach_topic -> DOCK_APPROACH (phase 2, below) -- see
      _on_start_approach().
  KILL: target frozen at whatever it was -- a universal safety stop,
  reachable from every other phase, but recoverable: resume_search_topic
  -> SEARCH_DESCEND again (re-armed from the AUV's current pose) -- see
  _on_resume_search().
  Every phase keeps publishing at publish_rate (there's no "stop" concept
  -- race_docking_control is a generic, mission-agnostic tracker that just
  tracks whatever desired_odometry says).

Phase 2 (DOCK_APPROACH) -- see _tick_dock_approach()/_compute_dock_tracking():
desired = the station dock point's pose in odom (EKF pose (+) raw
detection), offset up by dock_approach_hover_offset_z in plain odom Z;
actual = the AUV's own dock point pose in odom (EKF (+) dock_offset).
z1 = (actual - desired) / dock_z1_divisor while further than
dock_near_threshold away; once closer, z1 = the raw (undivided)
difference instead (full gain for the final approach). dot_p_d =
(actual - desired) / dock_velocity_divisor, always. beta = J^-1*(-H1*z1
+ dot_p_d); z2 = v_body - beta; Z1 = J^-1*z1, with J built from the AUV's
own current attitude. Once within dock_approach_pos_tol of the hover
point, the target freezes there for good ("stay there"). If the tag
hasn't been seen for dock_approach_tag_lost_timeout seconds -> phase 2b,
SEARCH_WHILE_APPROACHING, below.

Phase 2b (SEARCH_WHILE_APPROACHING) -- see
_enter_search_while_approaching(): reuses the exact SEARCH_DESCEND/
SEARCH_TURN staircase logic above, just parameterized with a smaller
step (search_while_approaching_descend_step) and a cap
(search_while_approaching_max_depth_delta) measured from the depth this
state was entered at (not the original search_start_z). Starts with a
turn, then descend/turn/descend/... until the tag reappears (any single
detection, no debounce needed) -> back to DOCK_APPROACH, or the cap is
reached with nothing found -> KILL.

Phase 3 (DOCKING) -- see _tick_docking(): manually triggered
(start_docking_topic, only while in DOCK_APPROACH), identical law via
the same _compute_dock_tracking(), just targeting the dock point itself
(z offset 0) instead of the hover point. abort_docking_topic goes back
to DOCK_APPROACH (only while in DOCKING).

Assumption: roll and pitch are 0 at mission start and never commanded
away from 0 during search (no pitching search maneuver here, unlike an
earlier version of this mission in race_auv_dock_search_mission).

Z1/z2 tracking-error law (_compute_and_publish_tracking_()): computed
here (mission), not in race_docking_control, which now only ever sees
Z1/z2 -- already expressed in body-frame units, so it needs no frame-
transform work of its own beyond the final tau->thruster allocation.
z1 = actual_pose(odom) - desired_pose(odom); beta = J^-1*(-H1*z1 -
desired_pose_dot); z2 = actual_body_twist - beta; Z1 = J^-1*z1, with
J^-1 = block_diag(R^T, T_theta^-1) built from the EKF's own odom
attitude.

Outputs (see README): desired_odometry, actual_odometry (both
nav_msgs/Odometry, pose+twist in odom_frame; twist is raw EKF body
twist), Z1/z2 (std_msgs/Float64MultiArray, the tracking-error law's
output, race_docking_control's actual tracking input),
orientation_error_deg (std_msgs/Float64MultiArray, monitoring only,
[roll, pitch, yaw] in degrees, actual - desired, raw/unscaled --
published every phase via _publish_orientation_error_deg()), and
mission_state (std_msgs/String). This node does no thruster/control work
at all -- that's race_docking_control's job.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from mvp_msgs.msg import ControlProcess
from mvp_msgs.srv import ChangeState
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float64MultiArray, String
from std_srvs.srv import SetBool  # added by maryam: controller enable/disable for kill

from . import frames
from .geometry_utils import (
    body_twist_to_odom_consistent,
    invert_euler_rate_transform,
    quaternion_error,
    quaternion_from_rpy,
    rotation_matrix_from_quaternion,
    rpy_from_quaternion,
    triangle_wave,
    wrap_to_pi,
    yaw_from_quaternion,
)


# Fixed 180 deg yaw offset composed onto the dock target's raw
# orientation in _compute_dock_tracking() (DOCK_APPROACH/DOCKING) --
# quaternion composition (quat_mul), not plain degree addition.
# Compensates for the station's dock_point-vs-auv_dock_point convention
# mismatch.
_DOCK_YAW_CORRECTION = quaternion_from_rpy(0.0, 0.0, 0.0 )
# math.pi / 2.0

class MissionState(Enum):
    SEARCH_DESCEND = 'SEARCH_DESCEND'
    SEARCH_TURN = 'SEARCH_TURN'
    HOLD = 'HOLD'
    # Phase 2. See _enter_dock_approach()/_tick_dock_approach(). No longer
    # entered automatically from HOLD -- see _on_start_approach().
    DOCK_APPROACH = 'DOCK_APPROACH'
    # Tag lost >= dock_approach_tag_lost_timeout seconds during
    # DOCK_APPROACH -- see _enter_search_while_approaching(). Reuses the
    # exact SEARCH_DESCEND/SEARCH_TURN staircase logic (smaller step,
    # capped relative to the depth this state was entered at, not the
    # mission's original search_start_z). Tag found -> back to
    # DOCK_APPROACH; cap reached with nothing found -> KILL.
    SEARCH_WHILE_APPROACHING = 'SEARCH_WHILE_APPROACHING'
    # Phase 3, manually triggered. See _enter_docking()/_tick_docking().
    DOCKING = 'DOCKING'
    # Renamed from ABORT -- now recoverable (see _on_resume_search()),
    # same spirit as mvp_helm's own 'kill' state: a universal safety stop
    # reachable from every other phase, but not necessarily final.
    KILL = 'KILL'
    # Added by maryam: manual, from any state except KILL. Rise to the start
    # depth first, then move horizontally (and turn) to the start x/y/yaw,
    # then hold there. See _enter_return_home()/_tick_return_home().
    RETURN_HOME = 'RETURN_HOME'


class SearchMissionNode(Node):

    def __init__(self) -> None:
        super().__init__('search_mission_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('odom_frame', 'race_auv/odom')
        self.declare_parameter('base_frame', 'race_auv/base_link')
        self.declare_parameter('ekf_odom_topic', '/race_auv/odometry/filtered')
        self.declare_parameter('dock_pose_topic', '/race_station/dock_point/pose')
        self.declare_parameter('ekf_stale_timeout', 2.0)

        # Monitoring only (phase 1 doesn't use this for control): the
        # fuser's dock_pose_topic measurement is base_link-relative; this
        # re-expresses it as auv_dock_frame-relative (station's dock point
        # relative to the AUV's *own* dock point, not its base_link) via a
        # static TF looked up once and cached, then republished for
        # monitoring in Foxglove/ros2 topic echo.
        self.declare_parameter('auv_dock_frame', 'race_auv/auv_dock_point')
        self.declare_parameter('tag_relative_pose_topic', 'tag_relative_pose')
        # Monitoring only: the station dock point's pose in odom
        # (EKF pose (+) the raw detection), republished every detection.
        self.declare_parameter('dock_point_odom_topic', 'dock_point_odom')

        # EKF yaw-drift correction: once HOLD first confirms, the station's
        # pose in odom is frozen as a trusted reference (dock_ref, "the
        # station is fixed"). Every detection after that, the AUV's
        # absolute pose is recomputed directly from dock_ref + the fresh
        # detection (bypassing the EKF's own possibly-drifted estimate)
        # and republished here for robot_localization to fuse as an
        # absolute-yaw correction (see pose1_config in
        # robot_localization_sim.yaml -- yaw only, x/y/z/roll/pitch left
        # to the existing sensors).
        self.declare_parameter('apriltag_absolute_pose_topic', 'apriltag_absolute_pose')
        self.declare_parameter('apriltag_absolute_yaw_covariance', 0.05)

        self.declare_parameter('search_descend_speed', 0.1)
        # Staircase descent: leg 0 targets search_first_descend_limit below
        # the start; leg k>=1 targets search_first_descend_limit +
        # k*search_descend_step, capped at search_max_depth (the leg whose
        # target lands on the cap is the last one -- no detection there
        # means KILL instead of another descend).
        self.declare_parameter('search_first_descend_limit', 1.0)
        self.declare_parameter('search_descend_step', 0.3)
        self.declare_parameter('search_max_depth', 3.0)
        self.declare_parameter('search_turn_rate_deg_s', 10.0)
        self.declare_parameter('tag_debounce_count', 10)
        self.declare_parameter('tag_detection_gap', 0.3)

        # Previous provisional-HOLD-lost-tag creep-back speed, as a fraction
        # of whichever rate was active on the axis being crept back on --
        # replaced by the reacquire-wobble params below. Kept, not deleted.
        self.declare_parameter('hold_creep_fraction', 0.1)

        # HOLD's tag-reacquire search motion: if the tag hasn't been seen
        # for hold_lost_timeout seconds, wobble depth +-hold_reacquire_
        # z_amplitude and yaw +-hold_reacquire_yaw_amplitude_deg around the
        # pose that last saw the tag (both as slow triangle waves), until
        # the tag is seen again, then snap back to steady HOLD there.
        self.declare_parameter('hold_lost_timeout', 1.0)
        self.declare_parameter('hold_reacquire_z_amplitude', 0.3)
        self.declare_parameter('hold_reacquire_z_speed', 0.03)
        self.declare_parameter('hold_reacquire_yaw_amplitude_deg', 10.0)
        self.declare_parameter('hold_reacquire_yaw_rate_deg_s', 2.0)

        # Phase 2: approach the station. Desired = the station dock point's
        # pose in odom (same quantity as dock_point_odom_topic), position
        # offset up by dock_approach_hover_offset_z in plain odom Z (not
        # rotated through the dock's own local orientation -- avoids the
        # earlier dock-frame sign ambiguity entirely). Actual = the AUV's
        # own dock point pose in odom (EKF (+) dock_offset). z1 = (actual -
        # desired) / dock_z1_divisor while further than dock_near_threshold
        # away; once closer, z1 = the raw (undivided) difference instead --
        # full gain for the final approach. dot_p_d = (actual - desired) /
        # dock_velocity_divisor, always. beta = J^-1*(-H1*z1 + dot_p_d);
        # z2 = v_body - beta; Z1 = J^-1*z1. Once within dock_approach_pos_tol
        # of the hover point, freeze the target there for good ("stay
        # there") instead of continuing to re-derive it from the live
        # (unfiltered) detection every tick.
        self.declare_parameter('dock_approach_hover_offset_z', 2.0)
        self.declare_parameter('dock_z1_divisor', 10.0)
        self.declare_parameter('dock_velocity_divisor', 15.0)
        self.declare_parameter('dock_near_threshold', 0.2)
        self.declare_parameter('dock_approach_pos_tol', 0.1)
        # DOCK_APPROACH-only override for the z axis (DOCKING keeps using
        # dock_z1_divisor/dock_velocity_divisor uniformly for all axes).
        self.declare_parameter('dock_approach_z1_z_divisor', 20.0)
        self.declare_parameter('dock_approach_velocity_z_divisor', 30.0)
        # Test-only, temporary -- may be removed later: when true,
        # DOCK_APPROACH ignores every divisor above -- z1 = the raw
        # relative pose itself, dot_p_d = 0 for every axis.
        self.declare_parameter('dock_approach_test_raw_z1_zero_velocity', False)

        # Phase 3: docking, manually triggered (not automatic) -- same law
        # as DOCK_APPROACH (_compute_dock_tracking(), same tolerance),
        # just targeting the dock point itself (z offset 0) instead of the
        # hover point. Its own separate divisor pair (decoupled from
        # DOCK_APPROACH's dock_z1_divisor/dock_velocity_divisor above) --
        # DOCKING also feeds this same divided-down z1 into the control
        # node's integral action (see docking_z1_divisor in
        # docking_control.yaml, which must be kept in sync with this one
        # by hand). start_docking_topic only does anything while in
        # DOCK_APPROACH; abort_docking_topic only does anything while in
        # DOCKING (goes back to DOCK_APPROACH, fresh).
        self.declare_parameter('docking_z1_divisor', 1.0)
        self.declare_parameter('docking_velocity_divisor', 5.0)
        self.declare_parameter('start_docking_topic', 'start_docking')
        self.declare_parameter('abort_docking_topic', 'abort_docking')

        # HOLD no longer auto-advances into DOCK_APPROACH -- see
        # _on_start_approach(). Mirrors start_docking_topic's pattern.
        self.declare_parameter('start_approach_topic', 'start_approach')
        # Recovers from KILL back into SEARCH_DESCEND -- see
        # _on_resume_search().
        self.declare_parameter('resume_search_topic', 'resume_search')
        # Added by maryam: manual emergency stop from any state -- see
        # _on_kill().
        self.declare_parameter('kill_topic', 'kill')
        # Added by maryam: RETURN_HOME -- see _enter_return_home().
        self.declare_parameter('return_home_topic', 'return_home')
        self.declare_parameter('return_home_speed', 0.05)          # m/s, horizontal
        self.declare_parameter('return_home_z_speed', 0.03)        # m/s, rise
        self.declare_parameter('return_home_yaw_rate_deg_s', 2.0)  # deg/s
        self.declare_parameter('return_home_tol', 0.05)            # m, arrival
        self.declare_parameter('return_home_xy_gain', 0.2)         # 1/s, x/y pull

        # SEARCH_WHILE_APPROACHING (see _enter_search_while_approaching()):
        # if the tag hasn't been seen for this many seconds while in
        # DOCK_APPROACH, switch into a recovery search -- the exact same
        # staircase logic as the original search (_tick_search_descend()/
        # _tick_search_turn()), just parameterized with a smaller step and
        # a cap measured from wherever this recovery started (not the
        # mission's original search_start_z). Tag reacquired at any point
        # (no debounce needed, unlike the initial search) -> straight back
        # to DOCK_APPROACH. Cap reached with nothing found -> KILL.
        self.declare_parameter('dock_approach_tag_lost_timeout', 180.0)
        self.declare_parameter('search_while_approaching_descend_step', 0.10)
        self.declare_parameter('search_while_approaching_max_depth_delta', 1.0)

        self.declare_parameter('publish_rate', 20.0)

        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._ekf_stale_timeout = float(self.get_parameter('ekf_stale_timeout').value)
        self._auv_dock_frame = self.get_parameter('auv_dock_frame').value
        self._apriltag_absolute_yaw_covariance = float(
            self.get_parameter('apriltag_absolute_yaw_covariance').value)

        self._search_descend_speed = float(self.get_parameter('search_descend_speed').value)
        self._search_first_descend_limit = float(
            self.get_parameter('search_first_descend_limit').value)
        self._search_descend_step = float(self.get_parameter('search_descend_step').value)
        self._search_max_depth = float(self.get_parameter('search_max_depth').value)
        self._search_turn_rate = math.radians(
            float(self.get_parameter('search_turn_rate_deg_s').value))
        self._tag_debounce_count = int(self.get_parameter('tag_debounce_count').value)
        self._tag_detection_gap = float(self.get_parameter('tag_detection_gap').value)
        self._hold_creep_fraction = float(self.get_parameter('hold_creep_fraction').value)

        self._hold_lost_timeout = float(self.get_parameter('hold_lost_timeout').value)
        self._hold_reacquire_z_amplitude = float(
            self.get_parameter('hold_reacquire_z_amplitude').value)
        self._hold_reacquire_z_speed = float(self.get_parameter('hold_reacquire_z_speed').value)
        self._hold_reacquire_yaw_amplitude = math.radians(
            float(self.get_parameter('hold_reacquire_yaw_amplitude_deg').value))
        self._hold_reacquire_yaw_rate = math.radians(
            float(self.get_parameter('hold_reacquire_yaw_rate_deg_s').value))

        self._dock_approach_hover_offset_z = float(
            self.get_parameter('dock_approach_hover_offset_z').value)
        self._dock_z1_divisor = float(self.get_parameter('dock_z1_divisor').value)
        self._dock_velocity_divisor = float(self.get_parameter('dock_velocity_divisor').value)
        self._dock_near_threshold = float(self.get_parameter('dock_near_threshold').value)
        self._dock_approach_pos_tol = float(self.get_parameter('dock_approach_pos_tol').value)
        self._dock_approach_z1_z_divisor = float(
            self.get_parameter('dock_approach_z1_z_divisor').value)
        self._dock_approach_velocity_z_divisor = float(
            self.get_parameter('dock_approach_velocity_z_divisor').value)
        self._docking_z1_divisor = float(self.get_parameter('docking_z1_divisor').value)
        self._docking_velocity_divisor = float(
            self.get_parameter('docking_velocity_divisor').value)
        self._dock_approach_tag_lost_timeout = float(
            self.get_parameter('dock_approach_tag_lost_timeout').value)
        self._swa_descend_step = float(
            self.get_parameter('search_while_approaching_descend_step').value)
        self._swa_max_depth_delta = float(
            self.get_parameter('search_while_approaching_max_depth_delta').value)
        self._dock_approach_test_raw_z1_zero_velocity = bool(
            self.get_parameter('dock_approach_test_raw_z1_zero_velocity').value)

        # Z1/z2 tracking-error law (moved here from race_docking_control --
        # mission now computes the full backstepping error terms itself and
        # hands control just Z1/z2; H2 stays a control-package param).
        # Order: [x, y, z, roll, pitch, yaw].
        self.declare_parameter('h1_diag', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        h1 = [float(v) for v in self.get_parameter('h1_diag').value]
        self._h1_pos = np.array(h1[0:3], dtype=float)
        self._h1_ang = np.array(h1[3:6], dtype=float)

        self._publish_rate = float(self.get_parameter('publish_rate').value)

        # ---------------- State ----------------
        self._phase = MissionState.SEARCH_DESCEND
        self._armed = False  # frozen entry point captured once EKF is available

        # Desired target, held/ramped depending on phase. roll/pitch (and
        # their rates) and vx/vy stay at 0 throughout phase 1/HOLD (see
        # module docstring assumption) -- only DOCK_ORIENT/DOCK_APPROACH
        # actually move them. This full 6-DOF set feeds both
        # _publish_desired() and _compute_and_publish_tracking_() uniformly
        # across every phase.
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_z = 0.0
        self._target_roll = 0.0
        self._target_pitch = 0.0
        self._target_yaw = 0.0
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vz = 0.0     # feedforward desired twist for this tick
        self._target_roll_rate = 0.0
        self._target_pitch_rate = 0.0
        self._target_yaw_rate = 0.0
        # Unused by the current single-phase DOCK_APPROACH design (it uses
        # now_s only); was per-tick dt for the previous two-phase ramps.
        self._last_tick_time: Optional[float] = None

        self._search_start_z = 0.0   # absolute depth reference all leg targets are measured from
        self._leg_index = 0          # which descend leg we're on (0-based)
        self._descend_target_z = 0.0
        self._leg_entry_z = 0.0
        self._leg_entry_yaw = 0.0
        self._leg_entry_time = 0.0
        # Which descend/turn sub-step is current -- set by _start_descend_
        # leg()/_start_turn_leg() regardless of which profile is active;
        # needed because SEARCH_WHILE_APPROACHING (unlike the original
        # search) uses the *same* MissionState value for both its descend
        # and turn sub-legs, so _on_publish_timer() can't tell them apart
        # from self._phase alone.
        self._active_leg_kind = 'descend'
        # The staircase parameters _leg_target_depth()/_is_last_leg()/
        # _start_descend_leg()/_start_turn_leg()/_tick_search_turn()
        # actually read -- set to the original search's own values by
        # _setup_initial_search_profile() (called from _arm()), or
        # overwritten with the recovery search's own values by
        # _enter_search_while_approaching(). This is what lets the exact
        # same staircase logic serve both uses.
        self._active_search_start_z = 0.0
        self._active_search_first_limit = 0.0
        self._active_search_step = 0.0
        self._active_search_max_depth = 0.0
        self._active_search_phase = MissionState.SEARCH_DESCEND
        self._active_turn_phase = MissionState.SEARCH_TURN
        self._active_search_on_exhausted = None

        self._tag_streak = 0
        self._last_tag_time: Optional[float] = None

        # HOLD is "provisional" until tag_debounce_count consecutive
        # detections confirm it (see _tick_hold()): if the tag is lost
        # before that, creep back slowly on whichever single axis (z or
        # yaw) was active when HOLD was entered, in the *reverse* of that
        # motion's direction, at hold_creep_fraction of its speed -- so it
        # sweeps back through where it last saw the tag without
        # overshooting past it again. Once confirmed, the pose is frozen
        # for good; losing the tag after that no longer does anything.
        self._hold_axis = 'z'
        self._hold_confirmed = False
        self._hold_creeping = False
        self._hold_creep_start_time = 0.0
        self._hold_creep_start_value = 0.0

        # HOLD's tag-reacquire search motion (see _tick_hold()). *_center
        # is the pose that last saw the tag -- what the wobble is centered
        # on and what it snaps back to once the tag is seen again.
        self._hold_reacquiring = False
        self._hold_reacquire_start_time = 0.0
        self._hold_z_center = 0.0
        self._hold_yaw_center = 0.0

        # Phase 2 (DOCK_APPROACH) -- see _tick_dock_approach(). Once the
        # hover point is reached, frozen here for good ("stay there").
        self._dock_approach_reached = False
        self._dock_approach_frozen_pos: Optional[tuple[float, float, float]] = None
        self._dock_approach_frozen_quat: Optional[tuple[float, float, float, float]] = None

        # Phase 3 (DOCKING) -- see _tick_docking(). Same "stay there once
        # reached" pattern, independent of DOCK_APPROACH's own.
        self._docking_reached = False
        self._docking_frozen_pos: Optional[tuple[float, float, float]] = None
        self._docking_frozen_quat: Optional[tuple[float, float, float, float]] = None

        self._ekf_pos: Optional[tuple[float, float, float]] = None
        self._ekf_quat: Optional[tuple[float, float, float, float]] = None
        self._ekf_linear_body: Optional[tuple[float, float, float]] = None
        self._ekf_angular_body: Optional[tuple[float, float, float]] = None
        self._ekf_last_time: Optional[float] = None

        # base_link -> auv_dock_frame static offset, looked up once via TF
        # and cached (it's a fixed URDF transform, never changes).
        self._dock_offset: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None
        # Latest raw fuser detection (station's dock point relative to
        # base_link, exactly as published) -- used for tag_relative_pose
        # monitoring (see _publish_tag_relative_pose()).
        self._latest_dock_base_pos: Optional[tuple[float, float, float]] = None
        self._latest_dock_base_quat: Optional[tuple[float, float, float, float]] = None

        # Station dock point's pose in odom, recorded on every fresh
        # detection (see _publish_dock_point_odom()) and held at its last
        # value otherwise -- what DOCK_APPROACH/DOCKING read their target
        # from.
        self._dock_point_odom_pos: Optional[tuple[float, float, float]] = None
        self._dock_point_odom_quat: Optional[tuple[float, float, float, float]] = None

        # EKF yaw-drift correction reference (see _enter_hold() /
        # _publish_apriltag_absolute_pose()): the station's pose in odom,
        # frozen once at the first HOLD confirmation, never touched again.
        self._dock_ref_pos: Optional[tuple[float, float, float]] = None
        self._dock_ref_quat: Optional[tuple[float, float, float, float]] = None

        # ---------------- Pub/Sub ----------------
        self._desired_pub = self.create_publisher(Odometry, 'desired_odometry', 10)
        self._actual_pub = self.create_publisher(Odometry, 'actual_odometry', 10)
        self._tag_relative_pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('tag_relative_pose_topic').value, 10)
        # Same data as tag_relative_pose's orientation, just human/Foxglove
        # -readable: [roll_deg, pitch_deg, yaw_deg]. The quaternion in
        # tag_relative_pose is the complete/authoritative value; this is a
        # monitoring convenience only.
        self._tag_relative_rpy_pub = self.create_publisher(
            Float64MultiArray, 'tag_relative_rpy_deg', 10)
        self._dock_point_odom_pub = self.create_publisher(
            PoseStamped, self.get_parameter('dock_point_odom_topic').value, 10)
        self._apriltag_absolute_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter('apriltag_absolute_pose_topic').value, 10)
        # Z1/z2 tracking-error law output: race_docking_control's only
        # tracking inputs now (it no longer computes z1/beta/z2 itself).
        # [x, y, z, roll, pitch, yaw].
        self._z1_pub = self.create_publisher(Float64MultiArray, 'Z1', 10)
        self._z2_pub = self.create_publisher(Float64MultiArray, 'z2', 10)
        # Monitoring only: the raw (unscaled, un-rotated) orientation
        # error -- actual - desired, in degrees [roll, pitch, yaw] --
        # straight from quaternion_error(), before Z1's R^T/T_theta^-1
        # rotation and (in _compute_dock_tracking()) before any divisor.
        # Published by both _compute_and_publish_tracking_() (search/
        # HOLD) and _compute_dock_tracking() (DOCK_APPROACH/DOCKING) --
        # active in every phase.
        self._orientation_error_deg_pub = self.create_publisher(
            Float64MultiArray, 'orientation_error_deg', 10)
        # Full-precision raw_diff (unscaled, un-rotated -- position in
        # metres, orientation in quaternion_error()'s raw radian-ish
        # units, same 6-vector orientation_error_deg is a degrees
        # conversion of). Consumed by docking_control_node's DOCKING-only
        # integral action -- see _compute_dock_tracking(). Only published
        # there (DOCK_APPROACH/DOCKING), unlike orientation_error_deg.
        self._raw_diff_pub = self.create_publisher(Float64MultiArray, 'raw_diff', 10)

        # Added by maryam: PID-guidance test path -- feeds mvp_helm's
        # bhv_direct_control behavior (NOT controller/process/set_point
        # directly -- mvp_helm arbitrates per-DOF priority across all its
        # active behaviors, e.g. bhv_surfacing can override regardless of
        # what this mission requests; publishing straight to mvp_control
        # would bypass that entirely). Published in this node's own
        # natural odom/base_link frame -- bhv_direct_control transforms it
        # into mvp_helm's configured frame (world_ned/cg_link) itself via
        # TF, so no manual rotation is needed here. Entirely independent
        # of docking_control_node's own Z1/z2/raw_diff path above --
        # harmless (unused) whenever mvp_helm isn't running.
        self.declare_parameter(
            'direct_control_setpoint_topic',
            '/race_auv/mvp_helm/bhv_direct_control/desired_setpoints')
        self.declare_parameter('helm_change_state_service', '/race_auv/mvp_helm/change_state')
        self._direct_control_pub = self.create_publisher(
            ControlProcess, self.get_parameter('direct_control_setpoint_topic').value, 10)
        self._change_state_client = self.create_client(
            ChangeState, self.get_parameter('helm_change_state_service').value)
        self._helm_docking_state_requested = False
        # Added by maryam: mvp_control's enable/disable service. KILL
        # disables the controller (zero thrust on its next loop) instead of
        # relying only on the helm 'kill' state, after which mvp_control
        # keeps following the last setpoint until its no-setpoint timeout.
        self.declare_parameter('controller_enable_service', '/race_auv/controller/set')
        self._controller_set_client = self.create_client(
            SetBool, self.get_parameter('controller_enable_service').value)
        # Added by maryam: the frame mvp_control actually controls (its
        # "current" pose is this frame's pose in world_ned). mvp_helm's
        # bhv_direct_control applies only the world transform (odom ->
        # world_ned) to the setpoint's pose -- never the child-frame
        # rotation -- so the setpoint must already describe this frame.
        # See _publish_direct_control_setpoint().
        self.declare_parameter('control_child_frame', 'race_auv/cg_link')
        self._control_child_frame = self.get_parameter('control_child_frame').value
        self._control_child_offset = None  # cached base_link -> control_child_frame
        # Added by maryam: PID target hold tolerances (DOCK_APPROACH/DOCKING)
        # -- see _publish_dock_pid_setpoint().
        self.declare_parameter('pid_target_pos_tol', 0.03)
        self.declare_parameter('pid_target_ang_tol_deg', 2.0)
        self._pid_target_pos_tol = float(self.get_parameter('pid_target_pos_tol').value)
        self._pid_target_ang_tol = math.radians(
            float(self.get_parameter('pid_target_ang_tol_deg').value))
        self._pid_held_target = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_pub = self.create_publisher(String, 'mission_state', state_qos)

        self._odom_sub = self.create_subscription(
            Odometry, self.get_parameter('ekf_odom_topic').value, self._on_odometry, 10,
        )
        self._dock_sub = self.create_subscription(
            PoseStamped, self.get_parameter('dock_pose_topic').value, self._on_dock_pose, 10,
        )
        # Manual triggers for phase 3 (DOCKING) -- see _on_start_docking()/
        # _on_abort_docking().
        self._start_docking_sub = self.create_subscription(
            Empty, self.get_parameter('start_docking_topic').value, self._on_start_docking, 10,
        )
        self._abort_docking_sub = self.create_subscription(
            Empty, self.get_parameter('abort_docking_topic').value, self._on_abort_docking, 10,
        )
        # HOLD -> DOCK_APPROACH, now manual -- see _on_start_approach().
        self._start_approach_sub = self.create_subscription(
            Empty, self.get_parameter('start_approach_topic').value, self._on_start_approach, 10,
        )
        # KILL -> SEARCH_DESCEND, recoverable -- see _on_resume_search().
        self._resume_search_sub = self.create_subscription(
            Empty, self.get_parameter('resume_search_topic').value, self._on_resume_search, 10,
        )
        # Added by maryam: any state -> KILL, manual emergency stop -- see
        # _on_kill().
        self._kill_sub = self.create_subscription(
            Empty, self.get_parameter('kill_topic').value, self._on_kill, 10,
        )
        # Added by maryam: any state (except KILL) -> RETURN_HOME.
        self._return_home_speed = float(self.get_parameter('return_home_speed').value)
        self._return_home_z_speed = float(self.get_parameter('return_home_z_speed').value)
        self._return_home_yaw_rate = math.radians(
            float(self.get_parameter('return_home_yaw_rate_deg_s').value))
        self._return_home_tol = float(self.get_parameter('return_home_tol').value)
        self._return_home_xy_gain = float(self.get_parameter('return_home_xy_gain').value)
        self._home_pos = None   # (x, y, z) odom, saved in _arm()
        self._home_yaw = 0.0
        self._return_home_step = 'rise'   # 'rise' -> 'move' -> 'arrived'
        self._return_home_last_time = 0.0
        self._return_home_sub = self.create_subscription(
            Empty, self.get_parameter('return_home_topic').value, self._on_return_home, 10,
        )

        # ---------------- Timer ----------------
        self._node_start_time = self._now_seconds()
        self._publish_timer = self.create_timer(1.0 / self._publish_rate, self._on_publish_timer)

        self._publish_state()
        self.get_logger().info(
            f"Search mission started: odom_frame={self._odom_frame}, phase={self._phase.value}"
        )

    # ------------------------------------------------------------------ time

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- callbacks

    def _on_odometry(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._ekf_pos = (p.x, p.y, p.z)
        self._ekf_quat = (q.x, q.y, q.z, q.w)
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        self._ekf_linear_body = (lv.x, lv.y, lv.z)
        self._ekf_angular_body = (av.x, av.y, av.z)
        self._ekf_last_time = self._now_seconds()

    def _on_dock_pose(self, msg: PoseStamped) -> None:
        # Monitoring only: republish the dock-to-dock relative pose (not
        # used for control in this phase, presence/timing is all that
        # matters for the state machine below).
        self._publish_tag_relative_pose(msg)

        p = msg.pose.position
        q = msg.pose.orientation
        self._latest_dock_base_pos = (p.x, p.y, p.z)
        self._latest_dock_base_quat = (q.x, q.y, q.z, q.w)
        self._publish_dock_point_odom()
        self._publish_apriltag_absolute_pose()

        now_s = self._now_seconds()
        if self._last_tag_time is not None and (now_s - self._last_tag_time) > self._tag_detection_gap:
            self._tag_streak = 0
        self._tag_streak += 1
        self._last_tag_time = now_s

        self.get_logger().info(
            f"Tag detected ({self._tag_streak}/{self._tag_debounce_count} consecutive)"
        )

        # SEARCH_WHILE_APPROACHING: reacquisition is immediate, no
        # debounce needed (unlike the initial search's -> HOLD path) --
        # we already trust the existing dock-point estimate, this is just
        # resuming it.
        if self._phase == MissionState.SEARCH_WHILE_APPROACHING:
            self.get_logger().info("Tag reacquired during SEARCH_WHILE_APPROACHING -> DOCK_APPROACH")
            self._enter_dock_approach()
            return

        # Debounce also required before entering HOLD at all (reverted from
        # a "first detection is enough" version -- kept below, commented,
        # not deleted). Since entry now only ever happens once the streak
        # has already reached tag_debounce_count, _enter_hold() marks the
        # pose confirmed immediately -- the separate provisional/creep-back
        # machinery in _tick_hold() stays in place (harmless) but won't
        # actually trigger under this gate, since there's no longer a
        # window where HOLD is entered but not yet confirmed.
        if self._tag_streak < self._tag_debounce_count:
            return
        # elif self._phase == MissionState.HOLD and not self._hold_confirmed:
        #     if self._tag_streak >= self._tag_debounce_count:
        #         self._hold_confirmed = True
        #         self._hold_creeping = False
        #         self.get_logger().info(
        #             f"HOLD pose confirmed/fixed after {self._tag_streak} consecutive detections"
        #         )
        if self._phase in (MissionState.SEARCH_DESCEND, MissionState.SEARCH_TURN):
            self._enter_hold()

    # ------------------------------------------------------------- monitoring

    def _ensure_dock_offset(self) -> bool:
        """Look up (once, then cache forever -- it's a fixed URDF transform)
        base_link -> auv_dock_frame.
        """
        if self._dock_offset is not None:
            return True
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._auv_dock_frame, rclpy.time.Time(),
            )
        except Exception:
            return False
        t = tf.transform.translation
        q = tf.transform.rotation
        self._dock_offset = ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        self.get_logger().info(
            f"Resolved {self._base_frame} -> {self._auv_dock_frame}: "
            f"translation=[{t.x:+.3f}, {t.y:+.3f}, {t.z:+.3f}]"
        )
        return True

    def _ensure_control_child_offset(self) -> bool:
        """Added by maryam: look up (once, then cache -- fixed URDF
        transform) base_link -> control_child_frame (mvp_control's
        controlled frame, cg_link).
        """
        if self._control_child_offset is not None:
            return True
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._control_child_frame, rclpy.time.Time(),
            )
        except Exception:
            return False
        t = tf.transform.translation
        q = tf.transform.rotation
        self._control_child_offset = ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        self.get_logger().info(
            f"Resolved {self._base_frame} -> {self._control_child_frame} (PID setpoint frame)")
        return True

    def _publish_tag_relative_pose(self, msg: PoseStamped) -> None:
        if not self._ensure_dock_offset():
            return
        dock_to_base = frames.invert(*self._dock_offset)
        p = msg.pose.position
        q = msg.pose.orientation
        pos, quat = frames.compose(*dock_to_base, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._auv_dock_frame
        out.pose.position.x = float(pos[0])
        out.pose.position.y = float(pos[1])
        out.pose.position.z = float(pos[2])
        out.pose.orientation.x = quat[0]
        out.pose.orientation.y = quat[1]
        out.pose.orientation.z = quat[2]
        out.pose.orientation.w = quat[3]
        self._tag_relative_pose_pub.publish(out)

        roll, pitch, yaw = rpy_from_quaternion(*quat)
        rpy_msg = Float64MultiArray()
        rpy_msg.data = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
        self._tag_relative_rpy_pub.publish(rpy_msg)

    def _publish_dock_point_odom(self) -> None:
        """The station dock point's pose in odom -- EKF pose (+) the raw
        detection, composed fresh from whichever detection just arrived
        and *recorded* here (self._dock_point_odom_pos/quat) as the
        current best estimate. Not filtered/smoothed. Also the single
        source DOCK_APPROACH/DOCKING read their target from -- see
        _tick_dock_approach()/_tick_docking() -- so that once the tag is
        lost, they hold the *last recorded* value instead of continuing
        to recompute it from the (still-moving) live EKF pose combined
        with stale detection data.
        """
        if self._ekf_pos is None or self._ekf_quat is None:
            return
        pos, quat = frames.compose(
            self._ekf_pos, self._ekf_quat,
            self._latest_dock_base_pos, self._latest_dock_base_quat,
        )
        self._dock_point_odom_pos = pos
        self._dock_point_odom_quat = quat
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._odom_frame
        out.pose.position.x = float(pos[0])
        out.pose.position.y = float(pos[1])
        out.pose.position.z = float(pos[2])
        out.pose.orientation.x = quat[0]
        out.pose.orientation.y = quat[1]
        out.pose.orientation.z = quat[2]
        out.pose.orientation.w = quat[3]
        self._dock_point_odom_pub.publish(out)

    def _publish_apriltag_absolute_pose(self) -> None:
        """EKF yaw-drift correction: the AUV's absolute pose in odom,
        computed from the frozen dock_ref reference (+) the fresh
        detection -- bypassing the EKF's own current (possibly-drifted)
        estimate entirely. Only published once dock_ref exists (i.e.
        after HOLD has confirmed at least once). robot_localization fuses
        only the yaw component (pose1_config in
        robot_localization_sim.yaml) -- x/y/z/roll/pitch covariances here
        are set large so they're effectively ignored even if config ever
        changes to include them.
        """
        if self._dock_ref_pos is None or self._latest_dock_base_pos is None:
            return
        pos, quat = frames.compose(
            self._dock_ref_pos, self._dock_ref_quat,
            *frames.invert(self._latest_dock_base_pos, self._latest_dock_base_quat),
        )
        out = PoseWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._odom_frame
        out.pose.pose.position.x = float(pos[0])
        out.pose.pose.position.y = float(pos[1])
        out.pose.pose.position.z = float(pos[2])
        out.pose.pose.orientation.x = quat[0]
        out.pose.pose.orientation.y = quat[1]
        out.pose.pose.orientation.z = quat[2]
        out.pose.pose.orientation.w = quat[3]
        cov = [0.0] * 36
        large = 1.0e6
        for i in range(5):  # x, y, z, roll, pitch
            cov[i * 6 + i] = large
        cov[5 * 6 + 5] = self._apriltag_absolute_yaw_covariance  # yaw
        out.pose.covariance = cov
        self._apriltag_absolute_pose_pub.publish(out)

    # ------------------------------------------------------------ transitions

    def _arm(self, now_s: float) -> None:
        """Freeze x/y/yaw, record the absolute depth reference all leg
        targets are measured from, and start the first SEARCH_DESCEND leg
        -- all from wherever the AUV is the moment the first EKF sample
        arrives.
        """
        self._target_x = self._ekf_pos[0]
        self._target_y = self._ekf_pos[1]
        self._target_yaw = yaw_from_quaternion(*self._ekf_quat)
        self._search_start_z = self._ekf_pos[2]
        # added by maryam: start point for RETURN_HOME
        self._home_pos = (float(self._ekf_pos[0]), float(self._ekf_pos[1]),
                          float(self._ekf_pos[2]))
        self._home_yaw = self._target_yaw
        # added by maryam: RETURN_HOME sets x/y velocity; search never does
        self._target_vx = 0.0
        self._target_vy = 0.0
        self.get_logger().info(
            f"Home (start point) saved: ({self._home_pos[0]:.2f}, {self._home_pos[1]:.2f}, "
            f"{self._home_pos[2]:.2f}, yaw={math.degrees(self._home_yaw):.1f} deg)")
        self._leg_index = 0
        self._setup_initial_search_profile()
        self._start_descend_leg(now_s)
        self._armed = True

    def _setup_initial_search_profile(self) -> None:
        """(Re)point the shared staircase logic (_leg_target_depth()/
        _is_last_leg()/_start_descend_leg()/_start_turn_leg()/
        _tick_search_turn()) at the original search's own parameters --
        see _enter_search_while_approaching() for the other profile these
        same functions can be pointed at. Called from _arm() and from
        _on_resume_search() (via _arm() again)."""
        self._active_search_start_z = self._search_start_z
        self._active_search_first_limit = self._search_first_descend_limit
        self._active_search_step = self._search_descend_step
        self._active_search_max_depth = self._search_max_depth
        self._active_search_phase = MissionState.SEARCH_DESCEND
        self._active_turn_phase = MissionState.SEARCH_TURN
        self._active_search_on_exhausted = lambda: self._enter_kill(
            f"reached search_max_depth ({self._search_max_depth:.2f} m) "
            f"without a tag detection"
        )

    def _leg_target_depth(self, leg_index: int) -> float:
        """Absolute depth (metres below _active_search_start_z) this leg
        descends to -- shared by the original search and
        SEARCH_WHILE_APPROACHING, parameterized via the _active_search_*
        attributes (see _setup_initial_search_profile()/
        _enter_search_while_approaching())."""
        depth = self._active_search_first_limit + leg_index * self._active_search_step
        return min(depth, self._active_search_max_depth)

    def _is_last_leg(self, leg_index: int) -> bool:
        return self._leg_target_depth(leg_index) >= self._active_search_max_depth - 1.0e-9

    def _start_descend_leg(self, now_s: float) -> None:
        self._phase = self._active_search_phase
        self._active_leg_kind = 'descend'
        self._leg_entry_z = self._ekf_pos[2]
        self._descend_target_z = self._active_search_start_z - self._leg_target_depth(self._leg_index)
        self._leg_entry_time = now_s
        self.get_logger().info(
            f"{self._phase.value} leg {self._leg_index} -> "
            f"{self._leg_target_depth(self._leg_index):.2f} m below start"
        )
        self._publish_state()

    def _start_turn_leg(self, now_s: float) -> None:
        self._phase = self._active_turn_phase
        self._active_leg_kind = 'turn'
        self._leg_entry_yaw = self._target_yaw
        self._leg_entry_time = now_s
        self.get_logger().info(
            f"{self._phase.value} leg {self._leg_index} turn "
            f"(no tag yet at {self._leg_target_depth(self._leg_index):.2f} m)"
        )
        self._publish_state()

    def _enter_hold(self) -> None:
        # Which axis was actively moving when the tag was seen -- that's
        # the one _tick_hold() creeps back on if the tag is subsequently
        # lost before the pose is confirmed.
        self._hold_axis = 'z' if self._phase == MissionState.SEARCH_DESCEND else 'yaw'
        self._phase = MissionState.HOLD
        # Confirmed immediately: entry itself now requires tag_debounce_count
        # consecutive detections (see _on_dock_pose()), so the separate
        # post-entry confirmation step is already satisfied by construction.
        self._hold_confirmed = True
        self._hold_creeping = False
        self._hold_reacquiring = False
        self._hold_z_center = self._target_z
        self._hold_yaw_center = self._target_yaw
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0
        # EKF yaw-drift correction reference: frozen once, the first time
        # HOLD confirms ("the station is fixed"). See
        # _publish_apriltag_absolute_pose().
        if (self._dock_ref_pos is None and self._ekf_pos is not None
                and self._ekf_quat is not None and self._latest_dock_base_pos is not None):
            self._dock_ref_pos, self._dock_ref_quat = frames.compose(
                self._ekf_pos, self._ekf_quat,
                self._latest_dock_base_pos, self._latest_dock_base_quat,
            )
            self.get_logger().info("Froze apriltag absolute-pose reference (dock_ref) at HOLD")
        self.get_logger().info(
            f"Tag detected ({self._tag_streak} consecutive) -> HOLD (confirmed) at "
            f"({self._target_x:.2f}, {self._target_y:.2f}, {self._target_z:.2f}, "
            f"yaw={math.degrees(self._target_yaw):.1f} deg)"
        )
        self._publish_state()
        # No longer auto-advances into DOCK_APPROACH -- HOLD now sits
        # indefinitely until _on_start_approach() triggers it manually.

    def _on_start_approach(self, _msg: Empty) -> None:
        if self._phase != MissionState.HOLD:
            self.get_logger().warn(
                f"start_approach ignored -- currently {self._phase.value}, not HOLD")
            return
        self._enter_dock_approach()

    def _enter_dock_approach(self) -> None:
        """Phase 3: approach the station. See _tick_dock_approach()."""
        self._phase = MissionState.DOCK_APPROACH
        self._dock_approach_reached = False
        self._dock_approach_frozen_pos = None
        self._dock_approach_frozen_quat = None
        self.get_logger().info("-> DOCK_APPROACH")
        self._publish_state()

    def _compute_dock_tracking(
            self, desired_pos: np.ndarray, desired_quat: tuple[float, float, float, float],
            z1_divisor: Optional[float] = None, velocity_divisor: Optional[float] = None,
            z1_z_divisor: Optional[float] = None, velocity_z_divisor: Optional[float] = None,
            test_raw_z1_zero_velocity: bool = False,
    ) -> Optional[tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]]:
        """Shared math for DOCK_APPROACH/DOCKING (identical law, only the
        target and base divisors differ): actual = the AUV's dock point
        pose in odom (EKF (+) dock_offset). z1 = (actual - desired) /
        z1_divisor (defaults to dock_z1_divisor -- DOCK_APPROACH's own;
        DOCKING passes docking_z1_divisor instead, its own separate,
        independently-tunable value -- see _tick_docking()). dot_p_d =
        (actual - desired) / velocity_divisor (defaults to
        dock_velocity_divisor, DOCKING passes docking_velocity_divisor),
        independent of z1's scaling. beta = J^-1*(-H1*z1 + dot_p_d);
        z2 = v_body - beta; Z1 = J^-1*z1, J built from the AUV's own
        current attitude. Publishes Z1/z2 and
        desired/actual odometry as a side effect. Returns (auv_pos,
        auv_quat, raw_diff_pos) -- raw_diff_pos is the *unscaled* position
        error, for the caller's own arrival check -- or None if the
        EKF/dock_offset aren't ready yet.

        z1_divisor/velocity_divisor: base divisor for all six axes; None
        (the default, used by DOCK_APPROACH) falls back to
        dock_z1_divisor/dock_velocity_divisor. DOCKING passes
        docking_z1_divisor/docking_velocity_divisor explicitly instead --
        its own separate, independently-tunable divisor, decoupled from
        DOCK_APPROACH's.

        z1_z_divisor/velocity_z_divisor: optional per-axis override for
        just the z component on top of the base divisor above (used by
        DOCK_APPROACH only, for a gentler/stronger z divisor than
        x/y/roll/pitch/yaw get); None (the default, used by DOCKING)
        means z uses the same (base) divisor as every other axis.

        test_raw_z1_zero_velocity: test-only, temporary -- may be removed
        later. When True, ignores every divisor above: z1 = the raw
        (undivided) relative pose itself, dot_p_d = 0 for every axis.
        """
        if (self._ekf_pos is None or self._ekf_quat is None
                or self._ekf_linear_body is None or self._ekf_angular_body is None
                or not self._ensure_dock_offset()):
            return None

        auv_pos, auv_quat = frames.compose(self._ekf_pos, self._ekf_quat, *self._dock_offset)

        # Compose the fixed 180 deg yaw offset onto the dock target's raw
        # orientation (see _DOCK_YAW_CORRECTION) before comparing.
        desired_quat = frames.quat_mul(desired_quat, _DOCK_YAW_CORRECTION)

        raw_diff_pos = np.asarray(auv_pos, dtype=float) - desired_pos
        raw_diff_ang = np.asarray(quaternion_error(desired_quat, auv_quat), dtype=float)
        raw_diff = np.concatenate([raw_diff_pos, raw_diff_ang])

        raw_diff_msg = Float64MultiArray()
        raw_diff_msg.data = [float(v) for v in raw_diff]
        self._raw_diff_pub.publish(raw_diff_msg)

        self._publish_orientation_error_deg(raw_diff_ang)

        # Previous: near-threshold switch to raw (undivided) z1 for full
        # gain on the final approach. Reverted -- always divide by
        # dock_z1_divisor now, regardless of distance. Kept, not deleted.
        # if float(np.linalg.norm(raw_diff_pos)) < self._dock_near_threshold:
        #     z1 = raw_diff
        # else:
        #     z1 = raw_diff / self._dock_z1_divisor
        if test_raw_z1_zero_velocity:
            # Test-only, temporary -- may be removed later: z1 = the raw
            # relative pose itself, no division; dot_p_d = 0 throughout.
            z1 = raw_diff
            dot_p_d = np.zeros(6)
        else:
            base_z1 = self._dock_z1_divisor if z1_divisor is None else z1_divisor
            z1_divisors = np.full(6, base_z1)
            if z1_z_divisor is not None:
                z1_divisors[2] = z1_z_divisor
            z1 = raw_diff / z1_divisors

            base_vel = self._dock_velocity_divisor if velocity_divisor is None else velocity_divisor
            vel_divisors = np.full(6, base_vel)
            if velocity_z_divisor is not None:
                vel_divisors[2] = velocity_z_divisor
            dot_p_d = raw_diff / vel_divisors
        z1_pos, z1_ang = z1[0:3], z1[3:6]
        dot_p_d_pos, dot_p_d_ang = dot_p_d[0:3], dot_p_d[3:6]

        auv_roll, auv_pitch, _auv_yaw = rpy_from_quaternion(*auv_quat)
        r = rotation_matrix_from_quaternion(*auv_quat)
        t_inv = invert_euler_rate_transform(auv_roll, auv_pitch)

        beta_pos = r.T @ (-self._h1_pos * z1_pos + dot_p_d_pos)
        beta_ang = t_inv @ (-self._h1_ang * z1_ang + dot_p_d_ang)

        v_body_pos = np.asarray(self._ekf_linear_body, dtype=float)
        v_body_ang = np.asarray(self._ekf_angular_body, dtype=float)
        z2_pos = v_body_pos - beta_pos
        z2_ang = v_body_ang - beta_ang

        Z1_pos = r.T @ z1_pos
        Z1_ang = t_inv @ z1_ang

        z1_msg = Float64MultiArray()
        z1_msg.data = [float(v) for v in (*Z1_pos, *Z1_ang)]
        self._z1_pub.publish(z1_msg)
        z2_msg = Float64MultiArray()
        z2_msg.data = [float(v) for v in (*z2_pos, *z2_ang)]
        self._z2_pub.publish(z2_msg)

        self._publish_dock_approach_desired_actual(desired_pos, desired_quat, auv_pos, auv_quat)

        # Previous PID path (commented by maryam): published the station
        # dock point's own target pose labeled as base_link's, and a
        # velocity with the wrong sign (+raw_diff points away).
        # desired_rpy = rpy_from_quaternion(*desired_quat)
        # desired_vel_body_xy = (r.T @ dot_p_d_pos)[0:2]
        # self._publish_direct_control_setpoint(desired_pos, desired_rpy, desired_vel_body_xy)

        # Added by maryam: PID-guidance path -- see
        # _publish_dock_pid_setpoint(). Same per-axis divisors as z1/dot_p_d
        # above (test mode: full step, zero velocity).
        if test_raw_z1_zero_velocity:
            z1_scale = np.ones(6)
            vel_scale = np.zeros(6)
        else:
            z1_scale = 1.0 / z1_divisors
            vel_scale = 1.0 / vel_divisors
        self._publish_dock_pid_setpoint(
            auv_pos, auv_quat, desired_pos, desired_quat, z1_scale, vel_scale)

        return auv_pos, auv_quat, raw_diff_pos

    def _enter_search_while_approaching(self) -> None:
        """Triggered from _tick_dock_approach() when the tag hasn't been
        seen for dock_approach_tag_lost_timeout seconds. Reuses the exact
        SEARCH_DESCEND/SEARCH_TURN staircase logic (_tick_search_descend()/
        _tick_search_turn(), via the shared _active_search_* profile),
        parameterized with search_while_approaching_descend_step and a
        cap (search_while_approaching_max_depth_delta) measured from the
        depth this state was entered at -- not the mission's original
        search_start_z. Starts with a turn (leg -1: no descend has
        happened yet), per spec, then descend/turn/descend/... in
        search_while_approaching_descend_step increments. Tag reacquired
        at any point -> back to DOCK_APPROACH (see _on_dock_pose()). Cap
        reached with nothing found -> KILL.
        """
        self._phase = MissionState.SEARCH_WHILE_APPROACHING
        self._active_search_start_z = self._ekf_pos[2]
        self._active_search_first_limit = self._swa_descend_step
        self._active_search_step = self._swa_descend_step
        self._active_search_max_depth = self._swa_max_depth_delta
        self._active_search_phase = MissionState.SEARCH_WHILE_APPROACHING
        self._active_turn_phase = MissionState.SEARCH_WHILE_APPROACHING
        self._active_search_on_exhausted = lambda: self._enter_kill(
            f"search_while_approaching: reached {self._swa_max_depth_delta:.2f} m "
            f"recovery cap without a tag detection"
        )
        self._leg_index = -1
        self.get_logger().warn(
            f"DOCK_APPROACH: tag lost for >= {self._dock_approach_tag_lost_timeout:.0f}s "
            f"-> SEARCH_WHILE_APPROACHING"
        )
        self._start_turn_leg(self._now_seconds())

    def _tick_dock_approach(self, now_s: float) -> None:
        if (self._last_tag_time is not None
                and (now_s - self._last_tag_time) >= self._dock_approach_tag_lost_timeout):
            self._enter_search_while_approaching()
            return
        if self._dock_approach_reached:
            desired_pos = np.asarray(self._dock_approach_frozen_pos, dtype=float)
            desired_quat = self._dock_approach_frozen_quat
        else:
            # Recorded pose (see _publish_dock_point_odom()) -- last fresh
            # detection composed with the EKF pose *at that instant*, held
            # as-is until the next detection updates it. Deliberately NOT
            # recomputed here from the live/current EKF pose, which would
            # let the AUV's current attitude leak into the target once the
            # tag is lost (the previous bug).
            if self._dock_point_odom_pos is None:
                return
            dock_pos, dock_quat = self._dock_point_odom_pos, self._dock_point_odom_quat
            desired_pos = np.array(
                [dock_pos[0], dock_pos[1], dock_pos[2] + self._dock_approach_hover_offset_z])
            desired_quat = dock_quat

        result = self._compute_dock_tracking(
            desired_pos, desired_quat,
            z1_z_divisor=self._dock_approach_z1_z_divisor,
            velocity_z_divisor=self._dock_approach_velocity_z_divisor,
            test_raw_z1_zero_velocity=self._dock_approach_test_raw_z1_zero_velocity,
        )
        if result is None:
            return
        _auv_pos, _auv_quat, raw_diff_pos = result

        dist = float(np.linalg.norm(raw_diff_pos))
        if dist < self._dock_approach_pos_tol and not self._dock_approach_reached:
            self._dock_approach_reached = True
            self._dock_approach_frozen_pos = tuple(desired_pos)
            self._dock_approach_frozen_quat = desired_quat
            self.get_logger().info("DOCK_APPROACH: hover point reached, staying there")

    def _on_start_docking(self, _msg: Empty) -> None:
        if self._phase != MissionState.DOCK_APPROACH:
            self.get_logger().warn(
                f"start_docking ignored -- currently {self._phase.value}, not DOCK_APPROACH")
            return
        self._enter_docking()

    def _on_abort_docking(self, _msg: Empty) -> None:
        if self._phase != MissionState.DOCKING:
            self.get_logger().warn(
                f"abort_docking ignored -- currently {self._phase.value}, not DOCKING")
            return
        self.get_logger().info("abort_docking -> DOCK_APPROACH")
        self._enter_dock_approach()

    def _enter_docking(self) -> None:
        """Phase 3, manually triggered (see _on_start_docking()). Same
        law as DOCK_APPROACH, just targeting the dock point itself
        (z offset 0) instead of the hover point.
        """
        self._phase = MissionState.DOCKING
        self._docking_reached = False
        self._docking_frozen_pos = None
        self._docking_frozen_quat = None
        self.get_logger().info("DOCK_APPROACH -> DOCKING")
        self._publish_state()

    def _tick_docking(self, now_s: float) -> None:
        if self._docking_reached:
            desired_pos = np.asarray(self._docking_frozen_pos, dtype=float)
            desired_quat = self._docking_frozen_quat
        else:
            # Recorded pose, same as _tick_dock_approach() -- see
            # _publish_dock_point_odom().
            if self._dock_point_odom_pos is None:
                return
            desired_pos = np.array(self._dock_point_odom_pos, dtype=float)  # z offset 0
            desired_quat = self._dock_point_odom_quat

        result = self._compute_dock_tracking(
            desired_pos, desired_quat,
            z1_divisor=self._docking_z1_divisor,
            velocity_divisor=self._docking_velocity_divisor,
        )
        if result is None:
            return
        _auv_pos, _auv_quat, raw_diff_pos = result

        dist = float(np.linalg.norm(raw_diff_pos))
        if dist < self._dock_approach_pos_tol and not self._docking_reached:
            self._docking_reached = True
            self._docking_frozen_pos = tuple(desired_pos)
            self._docking_frozen_quat = desired_quat
            self.get_logger().info("DOCKING: dock point reached, staying there")

    def _publish_orientation_error_deg(self, raw_diff_ang: np.ndarray) -> None:
        """Monitoring only: raw_diff_ang (actual - desired, straight from
        quaternion_error(), in radians) republished as [roll, pitch, yaw]
        in degrees -- before Z1's R^T/T_theta^-1 rotation and (in
        _compute_dock_tracking()) before any dock_z1_divisor scaling.
        """
        msg = Float64MultiArray()
        msg.data = [math.degrees(float(v)) for v in raw_diff_ang]
        self._orientation_error_deg_pub.publish(msg)

    # Added by maryam: PID-guidance test path (see __init__'s
    # _direct_control_pub/_change_state_client setup). Called from both
    # _compute_and_publish_tracking_() (search/HOLD) and
    # _compute_dock_tracking() (DOCK_APPROACH/DOCKING) -- after those
    # functions' own Z1/z2/raw_diff publishes, and wrapped defensively,
    # so a problem here can never prevent docking_control_node's own
    # inputs from going out.
    def _maybe_request_helm_docking_state(self) -> None:
        """Fire-and-forget, non-blocking: ask mvp_helm to switch into its
        "docking" FSM state (see helm_sim.yaml) so bhv_direct_control
        actually has priority and starts forwarding our setpoints to
        mvp_control. Only ever attempted once per node lifetime, and
        only once the service is actually available -- silently retries
        (no log spam) on ticks before mvp_helm has come up, and stays
        silent forever if mvp_helm/mvp_control aren't running at all
        (e.g. while testing docking_control_node instead).
        """
        if self._helm_docking_state_requested:
            return
        if not self._request_helm_state('docking'):
            return
        self._helm_docking_state_requested = True

    # commented by maryam: replaced by the version below, which can chain a
    # second state (needed because helm 'kill' only transitions to 'start',
    # so resuming must go kill -> start -> docking).
    # def _request_helm_state(self, state_name: str) -> bool:
    #     """Fire-and-forget, non-blocking mvp_helm change_state request.
    #     Returns False (does nothing else) if the service isn't currently
    #     available -- never blocks waiting for it. Safe to call with
    #     mvp_helm/mvp_control not running at all (e.g. while testing
    #     docking_control_node instead) -- just returns False every time.
    #     """
    #     if not self._change_state_client.service_is_ready():
    #         return False
    #     req = ChangeState.Request()
    #     req.state = state_name
    #     req.caller = self.get_name()
    #     future = self._change_state_client.call_async(req)
    #     future.add_done_callback(
    #         lambda f: self._on_change_state_response(f, state_name))
    #     return True

    # added by maryam: same as above, plus optional next_state requested
    # once this one is accepted.
    def _request_helm_state(self, state_name: str, next_state: Optional[str] = None) -> bool:
        """Fire-and-forget, non-blocking mvp_helm change_state request.
        Returns False (does nothing else) if the service isn't currently
        available -- never blocks waiting for it. Safe to call with
        mvp_helm/mvp_control not running at all (e.g. while testing
        docking_control_node instead) -- just returns False every time.
        If next_state is given, it is requested once state_name is accepted.
        """
        if not self._change_state_client.service_is_ready():
            return False
        req = ChangeState.Request()
        req.state = state_name
        req.caller = self.get_name()
        future = self._change_state_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_change_state_response(f, state_name, next_state))
        return True

    # added by maryam: enable/disable mvp_control (fire-and-forget, same
    # pattern as _request_helm_state()). Returns False if the service
    # isn't available (e.g. testing docking_control_node instead).
    def _request_controller_enabled(self, enabled: bool) -> bool:
        if not self._controller_set_client.service_is_ready():
            return False
        req = SetBool.Request()
        req.data = enabled
        future = self._controller_set_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_controller_set_response(f, enabled))
        return True

    # added by maryam
    def _on_controller_set_response(self, future, enabled: bool) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 -- never let this affect the mission
            self.get_logger().warn(f"controller/set({enabled}) call failed: {exc}")
            return
        self.get_logger().info(f"mvp_control: {resp.message}")

    # commented by maryam: replaced by the version below (adds next_state)
    # def _on_change_state_response(self, future, state_name: str) -> None:
    def _on_change_state_response(self, future, state_name: str,
                                  next_state: Optional[str] = None) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 -- never let this affect the mission
            self.get_logger().warn(f"mvp_helm change_state({state_name}) call failed: {exc}")
            return
        if resp.status:
            self.get_logger().info(f"mvp_helm switched to '{state_name}' state")
            # added by maryam: chain the next state (e.g. start -> docking)
            if next_state is not None:
                self._request_helm_state(next_state)
        else:
            self.get_logger().warn(f"mvp_helm rejected change_state to '{state_name}'")

    def _publish_dock_pid_setpoint(
            self, auv_pos: np.ndarray, auv_quat: tuple[float, float, float, float],
            desired_pos: np.ndarray, desired_quat: tuple[float, float, float, float],
            z1_scale: np.ndarray, vel_scale: np.ndarray,
    ) -> None:
        """Added by maryam: PID-guidance setpoint for DOCK_APPROACH/DOCKING.

        Step 1 -- dock point, odom frame. Target = the dock-point target
        itself, held fixed until it moves more than pid_target_pos_tol /
        pid_target_ang_tol_deg (mvp_control resets the integral on every
        setpoint change). Desired velocity is -raw_diff*vel_scale (same
        divisors as the RBF dot_p_d). (Updated by maryam: z1_scale drives the
        carrot again, desired = current - z1_scale*raw_diff, since mvp_control
        no longer zeroes the integral on setpoint change; hold tolerances are
        0 in the yaml.) Orientation uses the Euler-angle error (current rpy -
        target rpy, wrapped) instead of raw_diff's quaternion-based
        rotation vector, so the Euler-rate transform in step 2 is exact.

        Step 2 -- odom -> dock-point body axes at the current attitude
        (J^-1): R^T for linear velocity, T_theta^-1 for Euler rates.

        Step 3 -- dock point -> base_link (rigid body): pose = dock (+)
        inverse(dock_offset); angular velocity is the same for the whole
        body (only re-expressed in base axes); linear velocity
        v_base = v_dock - w x r_off, r_off = dock point position in
        base_link.
        """
        if self._dock_offset is None:
            return
        try:
            # Added by maryam: hold the PID target fixed -- mvp_control zeroes
            # an axis's integral whenever its setpoint changes, so a target
            # that moves every tick (the carrot, or the raw dock estimate's
            # detection jitter) kills the z integral and the AUV floats up.
            # Only accept a new target when it moves more than the tolerance.
            desired_pos = np.asarray(desired_pos, dtype=float)
            if self._pid_held_target is None:
                self._pid_held_target = (desired_pos, desired_quat)
            else:
                held_pos, held_quat = self._pid_held_target
                dot = abs(float(np.dot(held_quat, desired_quat)))
                ang = 2.0 * math.acos(min(1.0, dot))
                if (np.linalg.norm(desired_pos - held_pos) > self._pid_target_pos_tol
                        or ang > self._pid_target_ang_tol):
                    self._pid_held_target = (desired_pos, desired_quat)
            tgt_pos, tgt_quat = self._pid_held_target

            # Step 1: dock-point target pose + desired rates, odom frame.
            auv_pos = np.asarray(auv_pos, dtype=float)
            raw_diff_pos = auv_pos - tgt_pos
            # Previous (commented by maryam): moving carrot -- reset the PID
            # integral every tick.
            # carrot_dock_pos = auv_pos - z1_scale[0:3] * raw_diff_pos
            # added by maryam: carrot restored -- mvp_control no longer zeroes
            # the integral on setpoint change. desired = current - raw_diff/divisor.
            carrot_dock_pos = auv_pos - z1_scale[0:3] * raw_diff_pos
            v_dock_odom = -vel_scale[0:3] * raw_diff_pos

            cur_rpy = np.asarray(rpy_from_quaternion(*auv_quat), dtype=float)
            tgt_rpy = np.asarray(rpy_from_quaternion(*tgt_quat), dtype=float)
            e_rpy = np.array([wrap_to_pi(c - t) for c, t in zip(cur_rpy, tgt_rpy)])
            # carrot_dock_rpy = [wrap_to_pi(a) for a in cur_rpy - z1_scale[3:6] * e_rpy]
            # added by maryam: carrot restored (orientation)
            carrot_dock_rpy = [wrap_to_pi(a) for a in cur_rpy - z1_scale[3:6] * e_rpy]
            rpy_dot_dock = -vel_scale[3:6] * e_rpy

            # Step 2: odom -> dock-point body axes.
            r_dock = rotation_matrix_from_quaternion(*auv_quat)
            t_inv_dock = invert_euler_rate_transform(cur_rpy[0], cur_rpy[1])
            v_dock_body = r_dock.T @ v_dock_odom
            w_dock_body = t_inv_dock @ rpy_dot_dock

            # Step 3: dock point -> base_link.
            off_pos, off_quat = self._dock_offset
            # base_pos, base_quat = frames.compose(
            #     carrot_dock_pos, quaternion_from_rpy(*carrot_dock_rpy),
            #     *frames.invert(off_pos, off_quat),
            # )
            # commented by maryam: held-target pose, replaced by the carrot below
            # base_pos, base_quat = frames.compose(
            #     tgt_pos, tgt_quat, *frames.invert(off_pos, off_quat))
            # added by maryam: carrot restored
            base_pos, base_quat = frames.compose(
                carrot_dock_pos, quaternion_from_rpy(*carrot_dock_rpy),
                *frames.invert(off_pos, off_quat),
            )
            r_off = rotation_matrix_from_quaternion(*off_quat)  # dock axes -> base axes
            w_base_body = r_off @ w_dock_body
            v_base_body = (r_off @ v_dock_body
                           - np.cross(w_base_body, np.asarray(off_pos, dtype=float)))

            self._publish_direct_control_setpoint(
                base_pos, rpy_from_quaternion(*base_quat), v_base_body, w_base_body)
        except Exception as exc:  # noqa: BLE001 -- never let this affect docking_control_node
            self.get_logger().warn(
                f"dock PID setpoint failed: {exc}", throttle_duration_sec=5.0)

    def _publish_direct_control_setpoint(
            self, desired_pos: tuple[float, float, float],
            desired_rpy: tuple[float, float, float],
            desired_lin_vel_body: tuple[float, float, float],
            desired_ang_vel_body: tuple[float, float, float],
    ) -> None:
        """Builds and publishes mvp_msgs/ControlProcess to mvp_helm's
        bhv_direct_control, for mvp_control's controlled frame
        (control_child_frame, cg_link), with frame_id = odom.
        bhv_direct_control then only has to apply the world transform
        (odom -> world_ned) -- which is all it does to the pose.

        Inputs are base_link's desired state (callers already did the
        odom->body conversion: R^T for linear, T_theta^-1 for Euler rates):
        desired_pos/desired_rpy: base_link's desired pose in odom.
        desired_lin_vel_body/desired_ang_vel_body: base_link's desired
        twist in base_link's own body axes -- (u, v, w) and (p, q, r).
        Converted here to control_child_frame (rigid body): pose = base
        (+) offset; angular velocity re-expressed in the child axes;
        linear velocity transported to the child origin (v + w x r) and
        re-expressed in the child axes.
        """
        try:
            self._maybe_request_helm_docking_state()
            if not self._ensure_control_child_offset():
                self.get_logger().warn(
                    f"PID setpoint skipped: no TF {self._base_frame} -> "
                    f"{self._control_child_frame} yet", throttle_duration_sec=5.0)
                return
            off_pos, off_quat = self._control_child_offset
            child_pos, child_quat = frames.compose(
                desired_pos, quaternion_from_rpy(*desired_rpy), off_pos, off_quat)
            child_rpy = rpy_from_quaternion(*child_quat)
            r_base_to_child = rotation_matrix_from_quaternion(*off_quat).T
            w_base = np.asarray(desired_ang_vel_body, dtype=float)
            v_child = r_base_to_child @ (np.asarray(desired_lin_vel_body, dtype=float)
                                         + np.cross(w_base, np.asarray(off_pos, dtype=float)))
            w_child = r_base_to_child @ w_base

            msg = ControlProcess()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._odom_frame
            # Previous (commented by maryam): labeled base_link -- mvp_helm
            # never applied base_link -> cg_link to the orientation, so the
            # PID saw a ~180 deg roll error (cg_link is base_link flipped).
            # msg.child_frame_id = self._base_frame
            msg.child_frame_id = self._control_child_frame
            msg.control_mode = 'docking'
            msg.position.x = float(child_pos[0])
            msg.position.y = float(child_pos[1])
            msg.position.z = float(child_pos[2])
            msg.orientation.x = float(child_rpy[0])
            msg.orientation.y = float(child_rpy[1])
            msg.orientation.z = float(child_rpy[2])
            # Previous (commented by maryam): only u/v were filled.
            # msg.velocity.x = float(desired_vel_body_xy[0])
            # msg.velocity.y = float(desired_vel_body_xy[1])
            msg.velocity.x = float(v_child[0])
            msg.velocity.y = float(v_child[1])
            msg.velocity.z = float(v_child[2])
            msg.angular_rate.x = float(w_child[0])
            msg.angular_rate.y = float(w_child[1])
            msg.angular_rate.z = float(w_child[2])
            self._direct_control_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001 -- never let this affect docking_control_node
            self.get_logger().warn(
                f"direct_control setpoint publish failed: {exc}", throttle_duration_sec=5.0)

    def _publish_dock_approach_desired_actual(
            self, desired_pos: np.ndarray, desired_quat: tuple[float, float, float, float],
            auv_pos: tuple[float, float, float], auv_quat: tuple[float, float, float, float],
    ) -> None:
        """desired_odometry/actual_odometry for DOCK_APPROACH, plain odom
        frame (no twist -- dot_p_d is a heuristic feedforward baked
        straight into beta, not republished as a twist message here).
        """
        desired = Odometry()
        desired.header.stamp = self.get_clock().now().to_msg()
        desired.header.frame_id = self._odom_frame
        desired.child_frame_id = self._base_frame
        desired.pose.pose.position.x = float(desired_pos[0])
        desired.pose.pose.position.y = float(desired_pos[1])
        desired.pose.pose.position.z = float(desired_pos[2])
        desired.pose.pose.orientation.x = desired_quat[0]
        desired.pose.pose.orientation.y = desired_quat[1]
        desired.pose.pose.orientation.z = desired_quat[2]
        desired.pose.pose.orientation.w = desired_quat[3]
        self._desired_pub.publish(desired)

        actual = Odometry()
        actual.header.stamp = desired.header.stamp
        actual.header.frame_id = self._odom_frame
        actual.child_frame_id = self._base_frame
        actual.pose.pose.position.x = float(auv_pos[0])
        actual.pose.pose.position.y = float(auv_pos[1])
        actual.pose.pose.position.z = float(auv_pos[2])
        actual.pose.pose.orientation.x = auv_quat[0]
        actual.pose.pose.orientation.y = auv_quat[1]
        actual.pose.pose.orientation.z = auv_quat[2]
        actual.pose.pose.orientation.w = auv_quat[3]
        actual.twist.twist.linear.x = self._ekf_linear_body[0]
        actual.twist.twist.linear.y = self._ekf_linear_body[1]
        actual.twist.twist.linear.z = self._ekf_linear_body[2]
        actual.twist.twist.angular.x = self._ekf_angular_body[0]
        actual.twist.twist.angular.y = self._ekf_angular_body[1]
        actual.twist.twist.angular.z = self._ekf_angular_body[2]
        self._actual_pub.publish(actual)

    def _enter_kill(self, reason: str) -> None:
        """Renamed from _enter_abort() -- now recoverable, see
        _on_resume_search(). A universal safety stop: freezes the target,
        requests mvp_helm's own 'kill' state (zero-DOF, same guarantee as
        here)."""
        if self._phase == MissionState.KILL:
            return
        self._phase = MissionState.KILL
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0
        # added by maryam: RETURN_HOME sets x/y velocity too
        self._target_vx = 0.0
        self._target_vy = 0.0
        self.get_logger().warn(f"-> KILL: {reason}")
        self._publish_state()
        # PID-guidance test path: hand mvp_helm to its own 'kill' state;
        # reset the one-shot flag so a later resume can request 'docking'
        # again.
        self._request_helm_state('kill')
        self._helm_docking_state_requested = False
        # added by maryam: also disable mvp_control -> zero thrust right away
        self._request_controller_enabled(False)

    # added by maryam: manual emergency stop, works from any state. If
    # already in KILL, still re-sends the stop (helm kill + controller off).
    def _on_kill(self, _msg: Empty) -> None:
        if self._phase == MissionState.KILL:
            self.get_logger().warn("kill: already in KILL -- re-sending stop")
            self._request_helm_state('kill')
            self._request_controller_enabled(False)
            return
        self._enter_kill("manual kill (kill topic)")

    def _on_resume_search(self, _msg: Empty) -> None:
        # commented by maryam: also allowed once RETURN_HOME has arrived
        # if self._phase != MissionState.KILL:
        #     self.get_logger().warn(
        #         f"resume_search ignored -- currently {self._phase.value}, not KILL")
        #     return
        # added by maryam
        from_home = (self._phase == MissionState.RETURN_HOME
                     and self._return_home_step == 'arrived')
        if self._phase != MissionState.KILL and not from_home:
            self.get_logger().warn(
                f"resume_search ignored -- currently {self._phase.value}, "
                f"not KILL or RETURN_HOME (arrived)")
            return
        if self._ekf_pos is None or self._ekf_quat is None:
            self.get_logger().warn("resume_search ignored -- no EKF pose yet")
            return
        self.get_logger().info("resume_search -> SEARCH_DESCEND (re-armed)")
        # Reuses _arm() exactly -- re-captures x/y/yaw/search_start_z fresh
        # from wherever the AUV is right now, same as the original startup.
        self._arm(self._now_seconds())
        # added by maryam: from RETURN_HOME the helm/controller are already
        # running -- only KILL needs the restart below.
        if from_home:
            return
        # commented by maryam: helm 'kill' only transitions to 'start', so
        # requesting 'docking' directly from kill was rejected.
        # self._request_helm_state('docking')
        # added by maryam: re-enable mvp_control, then kill -> start -> docking
        self._request_controller_enabled(True)
        if self._request_helm_state('start', next_state='docking'):
            self._helm_docking_state_requested = True

    # ----------------------------------------------------------------- timer

    def _on_publish_timer(self) -> None:
        now_s = self._now_seconds()
        self._check_ekf_staleness(now_s)

        if not self._armed:
            if self._ekf_pos is None:
                return  # waiting for the first EKF sample to arm
            self._arm(now_s)

        if self._phase == MissionState.SEARCH_DESCEND:
            self._tick_search_descend(now_s)
        elif self._phase == MissionState.SEARCH_TURN:
            self._tick_search_turn(now_s)
        elif self._phase == MissionState.SEARCH_WHILE_APPROACHING:
            if self._active_leg_kind == 'descend':
                self._tick_search_descend(now_s)
            else:
                self._tick_search_turn(now_s)
        elif self._phase == MissionState.HOLD:
            self._tick_hold(now_s)
        elif self._phase == MissionState.DOCK_APPROACH:
            self._tick_dock_approach(now_s)
            return  # publishes its own desired/actual + Z1/z2
        elif self._phase == MissionState.DOCKING:
            self._tick_docking(now_s)
            return  # publishes its own desired/actual + Z1/z2
        # added by maryam
        elif self._phase == MissionState.RETURN_HOME:
            self._tick_return_home(now_s)
        # KILL: target is already frozen, nothing to update.

        self._publish_desired()
        self._publish_actual()
        # Full 6-DOF desired state -- roll/pitch/vx/vy/roll_rate/pitch_rate
        # stay 0 throughout phase 1 -- nothing else uses them right now,
        # kept generic so a future phase can reuse this same publish path.
        self._compute_and_publish_tracking_(
            desired_pos=(self._target_x, self._target_y, self._target_z),
            desired_rpy=(self._target_roll, self._target_pitch, self._target_yaw),
            desired_pos_dot=(self._target_vx, self._target_vy, self._target_vz),
            desired_rpy_dot=(self._target_roll_rate, self._target_pitch_rate, self._target_yaw_rate),
        )

    def _check_ekf_staleness(self, now_s: float) -> None:
        if self._phase == MissionState.KILL:
            return
        if self._ekf_last_time is None:
            if (now_s - self._node_start_time) > self._ekf_stale_timeout:
                self._enter_kill("EKF odometry never received")
            return
        age = now_s - self._ekf_last_time
        if age > self._ekf_stale_timeout:
            self._enter_kill(f"EKF odometry stale ({age:.2f}s > {self._ekf_stale_timeout:.2f}s)")

    def _tick_search_descend(self, now_s: float) -> None:
        elapsed = now_s - self._leg_entry_time
        dist = self._search_descend_speed * elapsed
        remaining = self._leg_entry_z - self._descend_target_z  # positive: target is deeper
        if dist >= remaining:
            self._target_z = self._descend_target_z
            self._start_turn_leg(now_s)
            return
        self._target_z = self._leg_entry_z - dist
        self._target_vz = -self._search_descend_speed

    def _tick_search_turn(self, now_s: float) -> None:
        elapsed = now_s - self._leg_entry_time
        turned = self._search_turn_rate * elapsed
        if turned >= 2.0 * math.pi:
            if self._is_last_leg(self._leg_index):
                self._active_search_on_exhausted()
            else:
                self._leg_index += 1
                self._start_descend_leg(now_s)
            return
        self._target_yaw = wrap_to_pi(self._leg_entry_yaw + turned)
        self._target_yaw_rate = self._search_turn_rate

    # Previous design: provisional HOLD (not yet debounce-confirmed) crept
    # back slowly on whichever single axis (z or yaw) was active when the
    # tag was first seen, if it was lost again before confirmation.
    # Replaced by the reacquire-wobble version below -- entry now always
    # requires tag_debounce_count consecutive detections already, so HOLD
    # starts "confirmed" by construction; this only mattered for the old
    # first-detection-triggers-HOLD design. Kept, not deleted.
    #
    # def _tick_hold(self, now_s: float) -> None:
    #     if self._hold_confirmed:
    #         self._target_vz = 0.0
    #         self._target_yaw_rate = 0.0
    #         return
    #
    #     lost = (self._last_tag_time is None) or (
    #         (now_s - self._last_tag_time) > self._tag_detection_gap)
    #
    #     if not lost:
    #         if self._hold_creeping:
    #             self._hold_creeping = False
    #             self.get_logger().info("Tag re-detected during HOLD -> holding here")
    #         self._target_vz = 0.0
    #         self._target_yaw_rate = 0.0
    #         return
    #
    #     if not self._hold_creeping:
    #         self._hold_creeping = True
    #         self._hold_creep_start_time = now_s
    #         self._hold_creep_start_value = (
    #             self._target_z if self._hold_axis == 'z' else self._target_yaw)
    #         self.get_logger().info(
    #             f"Lost tag during provisional HOLD -> creeping back on {self._hold_axis} "
    #             f"at {self._hold_creep_fraction:.2f}x the search rate"
    #         )
    #
    #     elapsed = now_s - self._hold_creep_start_time
    #     if self._hold_axis == 'z':
    #         creep_speed = self._hold_creep_fraction * self._search_descend_speed
    #         self._target_z = self._hold_creep_start_value + creep_speed * elapsed
    #         self._target_vz = creep_speed
    #         self._target_yaw_rate = 0.0
    #     else:
    #         creep_rate = self._hold_creep_fraction * self._search_turn_rate
    #         self._target_yaw = wrap_to_pi(self._hold_creep_start_value - creep_rate * elapsed)
    #         self._target_yaw_rate = -creep_rate
    #         self._target_vz = 0.0

    # Reacquire-wobble design (previous version of this method): if the tag
    # was lost, actively search nearby with a triangle-wave wobble in z/yaw
    # around the confirmed pose. Replaced by simply never moving the
    # target at all -- the desired pose stays exactly at the confirmed
    # sighting forever, so the tracking controller pulls the AUV straight
    # back there on its own if it drifts ("go back to the previous
    # state"), no active search motion needed. Kept, not deleted.
    #
    # def _tick_hold(self, now_s: float) -> None:
    #     lost = (self._last_tag_time is None) or (
    #         (now_s - self._last_tag_time) > self._hold_lost_timeout)
    #
    #     if not lost:
    #         if self._hold_reacquiring:
    #             self._hold_reacquiring = False
    #             self._target_z = self._hold_z_center
    #             self._target_yaw = self._hold_yaw_center
    #             self.get_logger().info("Tag reacquired during HOLD -> holding here")
    #         self._target_vz = 0.0
    #         self._target_yaw_rate = 0.0
    #         return
    #
    #     if not self._hold_reacquiring:
    #         self._hold_reacquiring = True
    #         self._hold_reacquire_start_time = now_s
    #         self.get_logger().info(
    #             "Tag lost during HOLD -> searching nearby "
    #             f"(z +-{self._hold_reacquire_z_amplitude:.2f} m, "
    #             f"yaw +-{math.degrees(self._hold_reacquire_yaw_amplitude):.0f} deg)"
    #         )
    #
    #     elapsed = now_s - self._hold_reacquire_start_time
    #     z_offset, vz = triangle_wave(
    #         elapsed, self._hold_reacquire_z_amplitude, self._hold_reacquire_z_speed)
    #     yaw_offset, yaw_rate = triangle_wave(
    #         elapsed, self._hold_reacquire_yaw_amplitude, self._hold_reacquire_yaw_rate)
    #     self._target_z = self._hold_z_center + z_offset
    #     self._target_yaw = wrap_to_pi(self._hold_yaw_center + yaw_offset)
    #     self._target_vz = vz
    #     self._target_yaw_rate = yaw_rate

    def _tick_hold(self, now_s: float) -> None:
        """HOLD is the terminal state of the search mission: the desired
        pose never moves from the pose that confirmed the tag sighting,
        tag visible or not. If the AUV has drifted off that pose (and so
        lost the tag), the tracking controller pulls it straight back
        there on its own -- no active search motion.
        """
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0

    # added by maryam: RETURN_HOME ------------------------------------------

    def _on_return_home(self, _msg: Empty) -> None:
        if self._phase in (MissionState.KILL, MissionState.RETURN_HOME):
            self.get_logger().warn(
                f"return_home ignored -- currently {self._phase.value}")
            return
        if (not self._armed or self._home_pos is None
                or self._ekf_pos is None or self._ekf_quat is None):
            self.get_logger().warn("return_home ignored -- no start point / EKF pose yet")
            return
        self._enter_return_home()

    def _enter_return_home(self) -> None:
        """Start from wherever the AUV is right now (EKF pose, roll/pitch
        0), rise to the start depth first, then move horizontally and turn
        to the start x/y/yaw (see _tick_return_home())."""
        self._phase = MissionState.RETURN_HOME
        self._return_home_step = 'rise'
        self._return_home_last_time = self._now_seconds()
        self._target_x = float(self._ekf_pos[0])
        self._target_y = float(self._ekf_pos[1])
        self._target_z = float(self._ekf_pos[2])
        self._target_roll = 0.0
        self._target_pitch = 0.0
        self._target_yaw = yaw_from_quaternion(*self._ekf_quat)
        self.get_logger().info(
            f"-> RETURN_HOME: rising to start depth {self._home_pos[2]:.2f} m first")
        self._publish_state()

    def _tick_return_home(self, now_s: float) -> None:
        """Ramps the target (same idea as the search ramps, so both the RBF
        and the PID path can follow it):
          'rise'    -- z only, at return_home_z_speed, x/y/yaw held.
          'move'    -- x/y in a straight line at return_home_speed, yaw
                       turns to the start yaw at return_home_yaw_rate.
          'arrived' -- target = start point, hold there.
        The PID path (helm 'teleop' mode) controls u/v, not x/y, so a
        x/y pull, return_home_xy_gain * (target - actual), is added to the
        desired x/y velocity in every step (capped at return_home_speed).
        """
        dt = max(0.0, now_s - self._return_home_last_time)
        self._return_home_last_time = now_s
        hx, hy, hz = self._home_pos
        ff_vx = ff_vy = 0.0
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0

        if self._return_home_step == 'rise':
            dz = hz - self._target_z
            step = self._return_home_z_speed * dt
            if abs(dz) <= step:
                self._target_z = hz
                self._return_home_step = 'move'
                self.get_logger().info("RETURN_HOME: start depth reached -> moving horizontally")
            else:
                sign = math.copysign(1.0, dz)
                self._target_z += sign * step
                self._target_vz = sign * self._return_home_z_speed

        elif self._return_home_step == 'move':
            dx, dy = hx - self._target_x, hy - self._target_y
            dist = math.hypot(dx, dy)
            step = self._return_home_speed * dt
            if dist <= step:
                self._target_x, self._target_y = hx, hy
            else:
                ux, uy = dx / dist, dy / dist
                self._target_x += ux * step
                self._target_y += uy * step
                ff_vx = ux * self._return_home_speed
                ff_vy = uy * self._return_home_speed
            dyaw = wrap_to_pi(self._home_yaw - self._target_yaw)
            ystep = self._return_home_yaw_rate * dt
            if abs(dyaw) <= ystep:
                self._target_yaw = self._home_yaw
            else:
                sign = math.copysign(1.0, dyaw)
                self._target_yaw = wrap_to_pi(self._target_yaw + sign * ystep)
                self._target_yaw_rate = sign * self._return_home_yaw_rate
            target_home = (self._target_x == hx and self._target_y == hy
                           and self._target_yaw == self._home_yaw)
            err = math.hypot(self._ekf_pos[0] - hx, self._ekf_pos[1] - hy)
            if target_home and err <= self._return_home_tol:
                self._return_home_step = 'arrived'
                self.get_logger().info(
                    f"RETURN_HOME: home reached ({err:.3f} m) -> holding there")

        # x/y pull toward the (ramped) target, see docstring.
        vx = ff_vx + self._return_home_xy_gain * (self._target_x - self._ekf_pos[0])
        vy = ff_vy + self._return_home_xy_gain * (self._target_y - self._ekf_pos[1])
        speed = math.hypot(vx, vy)
        if speed > self._return_home_speed > 0.0:
            vx *= self._return_home_speed / speed
            vy *= self._return_home_speed / speed
        self._target_vx = vx
        self._target_vy = vy

    # ------------------------------------------------------------------ output

    def _publish_desired(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = self._target_x
        msg.pose.pose.position.y = self._target_y
        msg.pose.pose.position.z = self._target_z
        # Previous: hardcoded roll/pitch=0. Generalized to the full 6-DOF
        # target fields, though roll/pitch/vx/vy/roll_rate/pitch_rate stay
        # at their 0.0 default throughout phase 1 -- nothing moves them.
        # qx, qy, qz, qw = quaternion_from_rpy(0.0, 0.0, self._target_yaw)
        qx, qy, qz, qw = quaternion_from_rpy(
            self._target_roll, self._target_pitch, self._target_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Feedforward desired twist for this phase (odom-consistent: linear
        # in odom frame, angular as roll/pitch/yaw rates) -- zero except
        # while actively ramping the target.
        # Previous: only z/yaw ever moved (phase 1/HOLD only).
        # msg.twist.twist.linear.z = self._target_vz
        # msg.twist.twist.angular.z = self._target_yaw_rate
        msg.twist.twist.linear.x = self._target_vx
        msg.twist.twist.linear.y = self._target_vy
        msg.twist.twist.linear.z = self._target_vz
        msg.twist.twist.angular.x = self._target_roll_rate
        msg.twist.twist.angular.y = self._target_pitch_rate
        msg.twist.twist.angular.z = self._target_yaw_rate
        self._desired_pub.publish(msg)

    def _publish_actual(self) -> None:
        if self._ekf_pos is None or self._ekf_quat is None:
            return
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = self._ekf_pos[0]
        msg.pose.pose.position.y = self._ekf_pos[1]
        msg.pose.pose.position.z = self._ekf_pos[2]
        msg.pose.pose.orientation.x = self._ekf_quat[0]
        msg.pose.pose.orientation.y = self._ekf_quat[1]
        msg.pose.pose.orientation.z = self._ekf_quat[2]
        msg.pose.pose.orientation.w = self._ekf_quat[3]

        # Previous: odom-consistent twist (odom-frame linear + Euler rates),
        # matching the old control-side tracking law. Replaced -- the RBF
        # input and the new Z1/z2 law both want raw body twist now. Kept,
        # not deleted.
        # if self._ekf_linear_body is not None and self._ekf_angular_body is not None:
        #     linear_odom, euler_rates = body_twist_to_odom_consistent(
        #         self._ekf_quat, self._ekf_linear_body, self._ekf_angular_body,
        #     )
        #     msg.twist.twist.linear.x = linear_odom[0]
        #     msg.twist.twist.linear.y = linear_odom[1]
        #     msg.twist.twist.linear.z = linear_odom[2]
        #     msg.twist.twist.angular.x = euler_rates[0]
        #     msg.twist.twist.angular.y = euler_rates[1]
        #     msg.twist.twist.angular.z = euler_rates[2]
        if self._ekf_linear_body is not None and self._ekf_angular_body is not None:
            msg.twist.twist.linear.x = self._ekf_linear_body[0]
            msg.twist.twist.linear.y = self._ekf_linear_body[1]
            msg.twist.twist.linear.z = self._ekf_linear_body[2]
            msg.twist.twist.angular.x = self._ekf_angular_body[0]
            msg.twist.twist.angular.y = self._ekf_angular_body[1]
            msg.twist.twist.angular.z = self._ekf_angular_body[2]
        self._actual_pub.publish(msg)

    def _compute_and_publish_tracking_(
        self,
        desired_pos: tuple[float, float, float],
        desired_rpy: tuple[float, float, float],
        desired_pos_dot: tuple[float, float, float],
        desired_rpy_dot: tuple[float, float, float],
    ) -> None:
        """Backstepping tracking-error law: computed here (mission), not in
        race_docking_control, which now only ever sees Z1/z2 -- already
        expressed in body-frame units, so it needs no frame-transform work
        of its own beyond the final tau->thruster allocation.

        z1 = actual_pose(odom) - desired_pose(odom)
        beta = J^-1 * (-H1*z1 - desired_pose_dot)
        z2 = actual_body_twist - beta
        Z1 = J^-1 * z1

        J^-1 = block_diag(R^T, T_theta^-1): R^T for the (orthogonal)
        rotation block, T_theta's true inverse (NOT transpose -- T_theta
        isn't orthogonal) for the Euler-rate block.
        """
        if (self._ekf_pos is None or self._ekf_quat is None
                or self._ekf_linear_body is None or self._ekf_angular_body is None):
            return

        roll, pitch, _yaw = rpy_from_quaternion(*self._ekf_quat)
        r = rotation_matrix_from_quaternion(*self._ekf_quat)
        t_inv = invert_euler_rate_transform(roll, pitch)

        z1_pos = np.asarray(self._ekf_pos, dtype=float) - np.asarray(desired_pos, dtype=float)
        q_desired = quaternion_from_rpy(*desired_rpy)
        z1_ang = np.asarray(quaternion_error(q_desired, self._ekf_quat), dtype=float)

        self._publish_orientation_error_deg(z1_ang)

        beta_pos = r.T @ (-self._h1_pos * z1_pos - np.asarray(desired_pos_dot, dtype=float))
        beta_ang = t_inv @ (-self._h1_ang * z1_ang - np.asarray(desired_rpy_dot, dtype=float))

        z2_pos = np.asarray(self._ekf_linear_body, dtype=float) - beta_pos
        z2_ang = np.asarray(self._ekf_angular_body, dtype=float) - beta_ang

        Z1_pos = r.T @ z1_pos
        Z1_ang = t_inv @ z1_ang

        z1_msg = Float64MultiArray()
        z1_msg.data = [float(v) for v in (*Z1_pos, *Z1_ang)]
        self._z1_pub.publish(z1_msg)

        z2_msg = Float64MultiArray()
        z2_msg.data = [float(v) for v in (*z2_pos, *z2_ang)]
        self._z2_pub.publish(z2_msg)

        # Added by maryam: PID-guidance path -- desired_pos/desired_rpy are
        # already base_link's own target here; the twist goes through J^-1
        # at the current attitude: R^T for the odom-frame linear velocity,
        # T_theta^-1 for the Euler-angle rates.
        # Previous (commented by maryam): only u/v were sent.
        # desired_vel_body_xy = (r.T @ np.asarray(desired_pos_dot, dtype=float))[0:2]
        # self._publish_direct_control_setpoint(desired_pos, desired_rpy, desired_vel_body_xy)
        lin_vel_body = r.T @ np.asarray(desired_pos_dot, dtype=float)
        ang_vel_body = t_inv @ np.asarray(desired_rpy_dot, dtype=float)
        self._publish_direct_control_setpoint(desired_pos, desired_rpy, lin_vel_body, ang_vel_body)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self._phase.value
        self._state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SearchMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
