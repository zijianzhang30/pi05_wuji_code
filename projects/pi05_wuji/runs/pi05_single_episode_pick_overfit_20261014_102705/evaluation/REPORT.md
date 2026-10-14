# π0.5 training-scene diagnostic
Snapshot Beijing: 2026-10-14T11:02:20.809013+08:00
Frozen step38000; pick_place train episode0 seed12001. This is a training-scene diagnostic, not a held-out success-rate estimate.

Pi05: success=False, reason=time_limit, object-contact frames=0, frames=450.
Expert endpoint comparator: success=True, reason=task_success, object-contact frames=114, frames=294.

Actual training-input teacher-forced joint errors (four noise draws at23 recorded observation frames):
- left_arm: first-action MAE 0.007262rad; full valid-chunk MAE 0.006485rad.
- left_hand: first-action MAE 0.002645rad; full valid-chunk MAE 0.002482rad.
- right_arm: first-action MAE 0.013737rad; full valid-chunk MAE 0.013813rad.
- right_hand: first-action MAE 0.004862rad; full valid-chunk MAE 0.004297rad.

Teacher-forced target-FK position differences: {"right_palm_link": {"chunk_mean_target_position_error_mm": 6.06119353986956, "chunk_p95_target_position_error_mm": 11.960582777331108, "first_mean_target_position_error_mm": 7.09040531493805, "first_p95_target_position_error_mm": 13.312305832013829}, "right_finger1_tip_link": {"chunk_mean_target_position_error_mm": 6.891267023790683, "chunk_p95_target_position_error_mm": 14.169121794319944, "first_mean_target_position_error_mm": 7.3632403364104935, "first_p95_target_position_error_mm": 16.352026092479992}, "right_finger2_tip_link": {"chunk_mean_target_position_error_mm": 7.730248132958566, "chunk_p95_target_position_error_mm": 15.659537593346514, "first_mean_target_position_error_mm": 8.969538765296882, "first_p95_target_position_error_mm": 18.393263152114805}}
Pi05 reset checks: {"qpos_max_abs_rad": 0.0, "qvel_max_abs_rad_s": 0.0, "gains_max_abs": 0.0, "palm_pose_max_abs": 0.0, "camera_matrix_max_abs": 0.0, "intrinsics_max_abs": 3.204619929420005e-06, "image_mean_abs_8bit": 2.033982204861111, "task_body_pose_max_abs": {"/World/TaskObjects/object": 4.5299530029296875e-06}}
Expert reset checks: {"qpos_max_abs_rad": 0.0, "qvel_max_abs_rad_s": 0.0, "gains_max_abs": 0.0, "palm_pose_max_abs": 0.0, "camera_matrix_max_abs": 0.0, "intrinsics_max_abs": 3.204619929420005e-06, "image_mean_abs_8bit": 2.0378114149305557, "task_body_pose_max_abs": {"/World/TaskObjects/object": 4.5299530029296875e-06}}

Pi05 measured palm positional difference versus demonstrated physical trajectory: {"mean": 181.2355194091797, "p95": 441.94366455078125, "precontact_mean": 48.68373107910156, "at_frame_150": 128.77767944335938, "at_frame_165": 159.6337890625}

Interpretation limits: one training scene and one closed-loop noise draw; offline prediction uses actual recorded RGB/qpos, live policy uses rendered RGB and its own state. Expert endpoint playback is a control, never a model success. Image mismatch must be assessed before attributing closed-loop failure entirely to policy fit. Effective contact/geometry criteria were not relaxed.
