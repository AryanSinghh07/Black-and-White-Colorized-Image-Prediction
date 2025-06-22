import cv2
import numpy as np
import os

def colorize_video(input_video, output_video):
    # Load model files
    prototxt = "models/colorization_deploy_v2.prototxt"
    model = "models/colorization_release_v2.caffemodel"
    points = "models/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Could not open input video")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to float and normalize
        img_rgb = frame.astype("float32") / 255.0
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
        L = img_lab[:, :, 0]

        # Resize and subtract 50 as required by model
        L_resized = cv2.resize(L, (224, 224))
        L_resized -= 50

        net.setInput(cv2.dnn.blobFromImage(L_resized))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (width, height))

        # Combine L and ab channels
        L = L[:, :, np.newaxis]
        lab_out = np.concatenate((L, ab), axis=2)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)
        bgr_out = (255 * bgr_out).astype("uint8")

        # Write frame
        out.write(bgr_out)

    cap.release()
    out.release()
    print(f"[✔] Video colorized and saved as: {output_video}")

    # Optional: Preview the colorized video
    cap_out = cv2.VideoCapture(output_video)
    print("Press 'q' to exit preview.")

    while cap_out.isOpened():
        ret, frame = cap_out.read()
        if not ret:
            break
        cv2.imshow("Colorized Video Preview", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap_out.release()
    cv2.destroyAllWindows()

    # Optional: Automatically open the video in your system's video player
    # os.startfile(output_video)  # Uncomment this on Windows

if __name__ == "__main__":
    input_video = "test.mp4"      # Replace with your grayscale input video
    output_video = "output.mp4"   # Output filename
    colorize_video(input_video, output_video)
