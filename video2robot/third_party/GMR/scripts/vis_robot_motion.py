from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument("--once", action="store_true",
                        help="Play motion once and exit")
    parser.add_argument("--loop", action="store_true",
                        help="Force loop playback (overrides --once)")
    parser.add_argument("--max_seconds", type=float, default=None,
                        help="Maximum playback/recording seconds (default: full clip when recording)")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)

    num_frames = len(motion_root_pos)
    # Recording normally expects one clip, so default to one-pass unless --loop is specified.
    play_once = (args.once or args.record_video) and (not args.loop)

    if args.max_seconds is not None:
        max_frames = max(1, int(round(float(args.max_seconds) * float(motion_fps))))
    elif play_once:
        max_frames = num_frames
    else:
        max_frames = None

    frame_idx = 0
    rendered = 0
    try:
        while True:
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )
            rendered += 1
            frame_idx += 1

            if max_frames is not None and rendered >= max_frames:
                break

            if frame_idx >= num_frames:
                if play_once:
                    break
                frame_idx = 0
    finally:
        env.close()