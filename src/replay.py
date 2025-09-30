import cv2
import os
import csv
import pandas as pd

def load_mistakes(mistakes_csv):
    """Load mistakes from CSV into list of dicts."""
    df = pd.read_csv(mistakes_csv)
    mistakes = df.to_dict(orient="records")
    return mistakes

def replay_video(video_path, mistakes, save_clips=False, save_full=False, save_advice_log=False, out_dir="replays/"):
    """
    Replay video with mistakes annotated and optional saving.

    Args:
        video_path (str): path to input video
        mistakes (list of dict): [{'frame': int, 'x': float, 'y': float, 'advice': str}]
        save_clips (bool): save short clips around mistakes
        save_full (bool): save full annotated replay
        save_advice_log (bool): save advice as CSV
        out_dir (str): output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare full video writer if needed
    full_writer = None
    if save_full:
        out_path = os.path.join(out_dir, os.path.basename(video_path).replace(".mp4", "_annotated.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        full_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Advice log
    advice_writer = None
    if save_advice_log:
        advice_file = os.path.join(out_dir, os.path.basename(video_path).replace(".mp4", "_advice_log.csv"))
        advice_writer = open(advice_file, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(advice_writer)
        csv_writer.writerow(["frame", "timestamp_sec", "x", "y", "advice"])

    mistake_frames = {int(m['frame']): m for m in mistakes}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in mistake_frames:
            m = mistake_frames[frame_idx]
            # Draw advice on video
            cv2.circle(frame, (int(m['x']), int(m['y'])), 25, (0,0,255), 3)
            cv2.putText(frame, m['advice'], (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2, cv2.LINE_AA)

            # Save advice log entry
            if advice_writer:
                timestamp = frame_idx / fps
                csv_writer.writerow([frame_idx, round(timestamp,2), int(m['x']), int(m['y']), m['advice']])

            # Save short clip
            if save_clips:
                clip_out = os.path.join(out_dir, f"{os.path.basename(video_path).replace('.mp4','')}_mistake_{frame_idx}.mp4")
                save_clip(video_path, frame_idx, fps, clip_out)

            # Pause for visibility
            cv2.imshow("Replay", frame)
            cv2.waitKey(2000)

        cv2.imshow("Replay", frame)

        if full_writer:
            full_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if full_writer:
        full_writer.release()
    if advice_writer:
        advice_writer.close()
        print(f"Advice log saved -> {advice_file}")
    cv2.destroyAllWindows()


def save_clip(video_path, mistake_frame, fps, out_path, pre_sec=2, post_sec=3):
    """Save short video clip around a mistake."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, mistake_frame - pre_sec * fps)
    end_frame   = min(total_frames-1, mistake_frame + post_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"Saved mistake clip -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--mistakes", required=True, help="Path to mistakes CSV from analyze_mistakes_ai.py")
    args = parser.parse_args()

    mistakes = load_mistakes(args.mistakes)

    while True:
        print("\nChoose saving option:")
        print("1. Show only (no saving)")
        print("2. Save mistake clips")
        print("3. Save full replay")
        print("4. Save advice log")
        print("5. Exit")
        print("👉 You can type multiple (e.g., '2 4')")
        choice = input("Your choice: ").strip().split()

        if "5" in choice:
            print("Exiting replay.")
            break

        save_clips = "2" in choice
        save_full = "3" in choice
        save_advice_log = "4" in choice

        replay_video(args.video, mistakes,
                     save_clips=save_clips,
                     save_full=save_full,
                     save_advice_log=save_advice_log)

        again = input("\nDo you want to re-run with different options? (y/n): ").lower()
        if again != "y":
            break
