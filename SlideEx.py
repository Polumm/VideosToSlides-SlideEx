import os
import pathlib
import time
import cv2
import imutils
import shutil
import img2pdf
import glob

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import mimetypes
import argparse


############# 定义常量

OUTPUT_SLIDES_DIR = f"./output"  # 默认输出文件夹

FRAME_RATE = 3  # 帧率：每秒需要处理的帧数越少，速度越快
WARMUP = FRAME_RATE  # 被跳过的初始帧数
FGBG_HISTORY = FRAME_RATE * 15  # 背景对象中的帧数
VAR_THRESHOLD = 16  # 方差阈值，用于判断当前像素是前景还是背景。一般默认为 16，如果光照变化明显，如阳光下的水面，建议设为 25，值越大灵敏度越低
DETECT_SHADOWS = False  # 是否检测影子，设为 True 为检测，False 为不检测，检测影子会增加程序时间复杂度，一般设置为 False
MIN_PERCENT = 0.1  # 在前景和背景之间的最小差值百分比，以检测运动是否已经停止
MAX_PERCENT = 3  # 在前景和背景之间的最大百分比的差异，以检测帧是否仍在运动

roi_defined = False
roi_points = [(0, 0), (0, 0)]
drawing = False


def select_roi(video_path, time_seconds):
    global roi_defined, roi_points

    # Reset the global variables
    roi_defined = False
    roi_points = [(0, 0), (0, 0)]

    # Get the frame at the specified time
    for frame_count, frame_time, frame in get_frames(video_path):
        if frame_time >= time_seconds:
            break

    frame = imutils.resize(frame, width=600)

    # Create a window to display the frame
    cv2.namedWindow("Select ROI")

    # Set the mouse callback function to handle ROI selection
    cv2.setMouseCallback("Select ROI", roi_selection)

    # Keep displaying the frame until the ROI is selected
    while not roi_defined:
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI, Press 'q' or 'ESC' to cancel ROI selection and exit", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' or 'ESC' to cancel ROI selection and exit
        if key == ord("q") or key == 27:
            roi_defined = False
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return roi_points


def show_ui():
    def open_video():
        video_path.set(filedialog.askopenfilename())

    def select_roi_button_click():
        select_roi(video_path.get(), float(time_seconds.get()))

    root = tk.Tk()
    root.title("ROI Selection")

    video_path = tk.StringVar()
    time_seconds = tk.StringVar(value="0")

    ttk.Label(root, text="Video path:").grid(column=0, row=0, sticky="w")
    ttk.Entry(root, textvariable=video_path).grid(column=1, row=0, sticky="we")
    ttk.Button(root, text="Browse", command=open_video).grid(column=2, row=0)

    ttk.Label(root, text="Time (seconds):").grid(column=0, row=1, sticky="w")
    ttk.Entry(root, textvariable=time_seconds).grid(column=1, row=1, sticky="we")

    ttk.Button(root, text="Select ROI", command=select_roi_button_click).grid(column=0, row=2, columnspan=3)

    root.columnconfigure(1, weight=1)
    root.mainloop()


def roi_selection(event, x, y, flags, param):
    global roi_points, drawing, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points[0] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_points[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_points[1] = (x, y)
        roi_defined = True


def get_frames(video_path):
    '''从位于 video_path 的视频返回帧的函数
    此函数跳过 FRAME_RATE 中定义的帧'''

    # 打开指向视频文件的指针初始化帧的宽度和高度
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')

    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = 0
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # 循环播放视频的帧
    while True:
        # 从视频中抓取一帧

        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)  # 将帧移动到时间戳
        frame_time += 1 / FRAME_RATE

        (_, frame) = vs.read()
        # 如果帧为None，那么我们已经到了视频文件的末尾
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()


def detect_unique_screenshots(video_path, output_folder_screenshot_path, roi):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=FGBG_HISTORY, varThreshold=VAR_THRESHOLD,
                                              detectShadows=DETECT_SHADOWS)

    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0
    for frame_count, frame_time, frame in get_frames(video_path):
        orig = frame.copy()  # clone the original frame (so we can save it later),
        frame = imutils.resize(frame, width=600)  # resize the frame

        # Apply ROI
        frame_roi = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        mask = fgbg.apply(frame_roi)  # apply the background subtractor

        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame
        if p_diff < MIN_PERCENT and not captured and frame_count > WARMUP:
            captured = True
            filename = f"{screenshoots_count:03}_{round(frame_time / 60, 2)}.png"

            path = str(pathlib.Path(output_folder_screenshot_path, filename))

            print("saving {}".format(path))
            cv2.imencode('.png', orig)[1].tofile(path)  # 防止imwrite中文乱码
            screenshoots_count += 1

        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= MAX_PERCENT:
            captured = False
    print(f'{screenshoots_count} screenshots Captured!')
    print(f'Time taken {time.time() - start_time}s')
    return


def initialize_output_folder(video_path):
    '''Clean the output folder if already exists'''
    (filesname, extension) = os.path.splitext(video_path)
    output_folder_name = video_path.rsplit(os.sep)[-1].replace(extension, '')
    output_folder_screenshot_path = os.path.join(OUTPUT_SLIDES_DIR, output_folder_name)

    if os.path.exists(output_folder_screenshot_path):
        shutil.rmtree(output_folder_screenshot_path)

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path



def convert_screenshots_to_pdf(output_folder_screenshot_path, video_path):
    (filesname, extension) = os.path.splitext(video_path)
    output_pdf_path = f"{video_path.rsplit('/')[-1].replace(extension, '')}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)


# if __name__ == "__main__":
#
#     input_dir_name = 'test'#要转换的视频所在的文件夹
#
#     for file_name in os.listdir(input_dir_name):
#         print(f"正在转换：{file_name}")
#         video_path = str(pathlib.Path(input_dir_name, file_name))
#         output_folder_screenshot_path = initialize_output_folder(video_path)
#         detect_unique_screenshots(video_path, output_folder_screenshot_path)
#
#         # 提取的图片转换为pdf
#         convert_screenshots_to_pdf(output_folder_screenshot_path)

def browse_input_directory():
    global input_dir_name
    input_dir_name = filedialog.askdirectory()
    if input_dir_name:
        input_dir_label.config(text=f"Selected input directory: {input_dir_name}")
    else:
        input_dir_label.config(text="No input directory selected")


def start_conversion():
    if input_dir_name:
        for file_name in os.listdir(input_dir_name):
            video_path = str(pathlib.Path(input_dir_name, file_name))

            # Check if the file is a video
            mimetype, _ = mimetypes.guess_type(video_path)
            if mimetype and mimetype.startswith('video'):
                print(f"正在转换：{file_name}")
                output_folder_screenshot_path = initialize_output_folder(video_path)

                # Get the desired time in seconds from the user, e.g. using an Entry widget or another method
                time_seconds = 45

                # Get ROI from the user based on the selected time
                roi = select_roi(video_path, time_seconds)

                # Check if the roi variable is not None before proceeding
                if roi is not None:
                    detect_unique_screenshots(video_path, output_folder_screenshot_path, roi)
                    convert_screenshots_to_pdf(output_folder_screenshot_path, video_path)
                else:
                    print("ROI selection canceled or failed. Skipping this video.")
            else:
                print(f"Skipping non-video file: {file_name}")
    else:
        input_dir_label.config(text="Please select an input directory first")


if __name__ == "__main__":
    input_dir_name = ''

    # Create a basic tkinter window
    root = tk.Tk()
    root.title("Video to PDF Converter")
    root.geometry("500x200")

    # Create a label and button for selecting the input directory
    input_dir_label = tk.Label(root, text="No input directory selected")
    input_dir_label.pack(pady=10)
    browse_button = tk.Button(root, text="Browse Input Directory", command=browse_input_directory)
    browse_button.pack(pady=10)

    # Create a button for starting the conversion
    start_button = tk.Button(root, text="Start Conversion", command=start_conversion)
    start_button.pack(pady=10)

    # Run the tkinter main loop
    root.mainloop()
