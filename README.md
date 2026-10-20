# race_docking_mission

Search + TAG_LOCKED mission node for the RACE AUV docking redesign. Part
of a two-package split with `race_docking_control`:

* **This package** owns everything frame/kinematics-related: reading the
  EKF, converting its raw body-frame twist into the odom-consistent
  representation the control law needs, generating the target
  pose/twist for whichever mission phase is active.
* **`race_docking_control`** is a generic, mission-agnostic backstepping +
  RBF-adaptive controller. It does no frame transforms at all except
  rotating its own output into body frame for the final thruster
  allocation step -- see that package's README.

## Mission logic (phase 1 only)

A staircase of (descend leg, turn leg) pairs, each descend leg going a
little deeper than the last, down to an absolute cap:

```
SEARCH_DESCEND(leg k) --(leg target reached, no tag)--> SEARCH_TURN(leg k)
      ^                                                       |
      |                          (360 deg, no tag, k < last)  |
      +-------------------------------------------------------+
      |
      |  (tag detected, any leg, either sub-phase)
      v                                                       |
    HOLD                            (360 deg, no tag, k == last leg) -> ABORT
```

Leg `k`'s target depth is `search_first_descend_limit +
k*search_descend_step`, capped at `search_max_depth` -- default legs land
at 1.0, 1.5, 2.0, 2.5, 3.0 m below the mission's starting depth. All leg
targets are measured from that one fixed starting depth (recorded once at
arm), not from wherever the previous leg happened to end -- a straight
staircase down, not a drifting spiral.

* **SEARCH_DESCEND** -- x/y/yaw held at the mission's starting pose. Sinks
  straight down (z decreasing, z-up frame) at `search_descend_speed` to
  the current leg's target depth, watching for an AprilTag detection the
  whole time.
* **SEARCH_TURN** -- x/y/z held at wherever the descend left off; yaw
  rotates a full 360 deg at `search_turn_rate_deg_s`, still watching for a
  detection. No detection by the end: if this wasn't the leg at
  `search_max_depth`, the leg index increments and a fresh
  `SEARCH_DESCEND` starts (one leg deeper); if it *was* the
  `search_max_depth` leg, `ABORT` instead.
* **HOLD** -- a tag was detected (debounced over `tag_debounce_count`
  consecutive fuser messages), in either sub-phase. Target freezes at
  whatever it was the instant of detection -- position *and* orientation
  both stay exactly where they were, nothing further is commanded -- then
  the mission immediately moves into **TAG_LOCKED** (see below).
* **ABORT** -- completed the `search_max_depth` leg's turn with no
  detection. Same "freeze and keep publishing" behavior as HOLD, differing
  only in the logged/reported `mission_state` -- see the note on why below.

## Mission logic (phase 2: TAG_LOCKED)

Entered automatically once `HOLD` confirms. Unlike phase 1, everything
here is planned and actuated relative to **the station's own dock point
frame `D`**, not `odom_frame` -- because the station doesn't move, `D` is
just as good a fixed/inertial reference as odom is, and choosing it means
the desired path and the vision-derived actual pose live in the same
frame without needing odom-composition smoothing.

1. **`D` is established once, on `TAG_LOCKED` entry, and never touched
   again** ("the station is fixed"): `dock_in_odom` = the EKF pose
   composed with the freshest fuser detection available at that instant.
   `d_from_odom = frames.invert(dock_in_odom)` is what re-expresses any
   odom-frame pose in `D` from then on.
2. **Desired path**: a receding-horizon quintic (replanned every
   `tag_locked_replan_period`, same C2-continuous approach as any
   quintic-based mission in this workspace) drives x/y/z/yaw from the AUV
   dock point's current pose in `D` toward the fixed hover point
   `(0, 0, tag_locked_target_z, 0, 0, 0)`. Roll/pitch are held at 0
   directly -- no trajectory needed, `race_docking_control` regulates any
   small deviation on its own.
3. **Actual pose/twist**, both in `D`, come from the EKF -- continuous,
   low-noise, already-fused IMU+DVL+depth -- re-expressed in `D`, **not**
   from differentiating consecutive vision poses (see "Why not
   differentiate vision poses for velocity" below):
   1. `auv_dock_point`'s pose in odom = EKF pose (+) the static
      `base_frame -> auv_dock_frame` offset.
   2. Re-express that in `D` via `d_from_odom`.
   3. `auv_dock_point`'s twist in body axes = EKF's raw body twist,
      lever-arm-corrected (`geometry_utils.transport_linear_velocity()`,
      rigid-body transport theorem) for the same offset. Angular velocity
      needs no such correction -- it's the same for every point on a
      rigid body.
   4. The linear part rotates straight into `D`. The angular part
      converts to Euler **rates** in `D` (`geometry_utils.
      euler_rate_transform()`, reusing the raw body angular velocity
      unchanged -- only the roll/pitch fed into the transform matrix come
      from `D` -- **not** a plain rotation of the angular velocity vector,
      since what pairs correctly with the quaternion-based orientation
      error is Euler rates, not raw `omega`).
4. `TAG_LOCKED` **never transitions away**. Once the hover point is
   reached, each replan just keeps re-targeting the same fixed point
   (near-zero displacement, `min_T`-duration quintics), so it degenerates
   naturally into station-keeping. "Hover reached and held" is logged once
   (after `tag_locked_hover_hold_time` inside tolerance) but doesn't
   change `mission_state`.

### Why not differentiate vision poses for velocity

Vision (AprilTag) pose estimates carry real per-frame jitter, and the
fuser only runs at ~5 Hz with gaps. Finite-differencing two noisy, low-rate
samples *amplifies* that noise (dividing position noise by a small `dt`),
which would feed directly into `z2` and then into `tau` -- fighting the
slew-rate limiter `race_docking_control` already has to smooth exactly
that kind of chatter. The EKF already solves this properly (that's what a
Kalman filter fusing IMU+DVL+depth is *for*): continuous, high-rate,
low-noise. So position comes from vision (that's the point of
`TAG_LOCKED`), but velocity still comes from the EKF, just re-expressed in
`D` per the steps above.

**Assumption**: roll and pitch are 0 at mission start and never commanded
away from 0 in this phase -- no pitching search maneuver here (unlike an
earlier design in `race_auv_dock_search_mission`), just heave then yaw.

**Why HOLD and ABORT both keep publishing** (rather than "stopping"):
`race_docking_control` is a generic tracker with no concept of mission
phases or success/failure -- it just continuously tracks whatever
`desired_odometry` says. "Stay right here" therefore has to mean "keep
publishing the same fixed target", not "stop publishing" (which would
leave the control package tracking a target that's aged out with no
defined behavior). `race_docking_control` does have its own independent
safety net for this: if `desired_odometry` itself goes stale (this node
crashes, etc.), it zeros the thrusters rather than holding forever.

## ROS interface

Outputs (topics are relative; both nodes share the `race_auv` namespace by
default, so no remapping is needed to wire them together):

* `desired_odometry` / `actual_odometry` (`nav_msgs/Odometry`,
  `publish_rate` Hz) -- **the same two topics carry both phases' data**;
  `race_docking_control` doesn't need to know or care which phase is
  active, since it just tracks whatever these say.
  - **Phase 1**: `frame_id=odom_frame`, `child_frame_id=base_frame`.
    Desired twist is a feedforward term (`linear.z = -search_descend_speed`
    during SEARCH_DESCEND, `angular.z = search_turn_rate` during
    SEARCH_TURN, zero otherwise); actual pose is the EKF's pose passed
    through unchanged, actual twist is the EKF's raw body twist
    **converted** to the odom-consistent representation (odom-frame linear
    velocity + roll/pitch/yaw *rates*) via
    `geometry_utils.body_twist_to_odom_consistent()` -- ported from
    `race_controller.cpp`'s `build_odom_state_()`/`euler_rate_transform_()`
    (read as reference; `simple_controller_pkg` itself is untouched).
  - **Phase 2 (TAG_LOCKED)**: `frame_id=tag_locked_frame_label` (a label,
    not a live broadcast TF frame), `child_frame_id` is still
    `base_frame` (needed downstream for thruster-allocation TF lookups,
    independent of which frame the pose itself is in). Both desired and
    actual pose/twist are expressed in the station's fixed dock frame `D`
    -- see "Mission logic (phase 2)" above for the full derivation.
* `mission_state` (`std_msgs/String`, latched) -- one of `SEARCH_DESCEND`,
  `SEARCH_TURN`, `HOLD`, `TAG_LOCKED`, `ABORT`.
* `tag_relative_pose_topic` (default `tag_relative_pose`,
  `geometry_msgs/PoseStamped`, `frame_id=auv_dock_frame`) -- monitoring
  only, not used for control in this phase. Republishes every fuser
  detection re-expressed relative to the AUV's own dock point instead of
  `base_link` (`frames.invert()` on a static `base_frame -> auv_dock_frame`
  TF, looked up once and cached, then `frames.compose()` with the
  detection).

Inputs:

* EKF odometry (`nav_msgs/Odometry`, param `ekf_odom_topic`, default
  `/race_auv/odometry/filtered`).
* AprilTag fuser dock pose (`geometry_msgs/PoseStamped`, param
  `dock_pose_topic`, default `/dock_point/pose`) -- its arrival timing is
  what drives the state machine (debounce count/gap; phase 1 just needs to
  know *whether* a tag is visible), but its actual pose value is also
  re-published on `tag_relative_pose_topic` for monitoring (see above).
* Static TF `base_frame -> auv_dock_frame` (default `race_auv/base_link ->
  race_auv/auv_dock_point`) -- looked up once via `tf2_ros`, cached
  forever. Used both for the `tag_relative_pose_topic` monitoring output
  and, in TAG_LOCKED, to compose the AUV dock point's actual pose/twist.

## Structure

* `search_mission_node.py` -- the mission node (ROS-dependent).
* `quintic.py` -- closed-form quintic segment math (no ROS dependency,
  unit tested in `test/test_quintic.py`), copied verbatim from
  `race_auv_dock_search_mission` (not modified there or here): used for
  TAG_LOCKED's receding-horizon path.
* `geometry_utils.py` -- pure math (no ROS dependency, unit tested in
  `test/test_geometry_utils.py`): quaternion/RPY helpers, the
  body-twist -> odom-consistent conversion, the rigid-body transport-
  theorem lever-arm correction (`transport_linear_velocity()`), and the
  Euler-rate/rotation-matrix helpers TAG_LOCKED's actual-state pipeline
  uses -- all of it is what keeps `race_docking_control` frame-transform-
  free in both phases.
* `frames.py` -- SE(3) pose composition (`compose()`) and inversion
  (`invert()`), used for the `tag_relative_pose_topic` monitoring output
  and for establishing/using `dock_in_odom`/`d_from_odom` in TAG_LOCKED.
* `config/search_mission.yaml`, `launch/search_mission.launch.py`.
