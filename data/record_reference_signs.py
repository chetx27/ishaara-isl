"""
record_reference_signs.py

Utility script to record a personal reference library of ISL signs for 
the lookup-and-stitch text-to-ISL generation (Phase 6).

Requires a webcam. Records to raw_videos/reference/<gloss>.mp4.
"""

import cv2
import os
import time
import argparse

def main(output_dir, gloss):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{gloss}.mp4")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"\nRecording sign for gloss: '{gloss}'")
    print("Press 'r' to start recording.")
    print("Press 'q' to stop recording and save.")

    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display instructions on frame
        display_frame = frame.copy()
        if not recording:
            cv2.putText(display_frame, f"Ready: {gloss}. Press 'r' to start.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"Recording: {gloss}. Press 'q' to stop.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)

        cv2.imshow('Recording Reference Sign', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording:
            recording = True
            print("Recording started...")
        elif key == ord('q'):
            if recording:
                print("Recording stopped. Saved to", out_path)
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record reference ISL signs.")
    parser.add_argument("--gloss", type=str, required=True, help="The gloss (word) to record.")
    parser.add_argument("--dir", type=str, default="reference_videos", help="Output directory.")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(data_dir, args.dir)
    
    main(output_dir, args.gloss)
