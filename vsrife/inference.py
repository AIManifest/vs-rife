from __future__ import annotations

import time
import math
import os
import cv2
import warnings
from fractions import Fraction
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import skvideo.io
import _thread
from queue import Queue, Empty
from .ssim import ssim_matlab

from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
# import vapoursynth as vs
print('INITIALIZING ENGINE')

__version__ = "5.2.0"
os.environ["CI_BUILD"] = "1"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "At pre-dispatch tracing")
warnings.filterwarnings("ignore", "Attempted to insert a get_attr Node with no underlying reference")
warnings.filterwarnings("ignore", "Node _run_on_acc_0_engine target _run_on_acc_0_engine _run_on_acc_0_engine of")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

models = [
    "4.0",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "4.10",
    "4.11",
    "4.12",
    "4.12.lite",
    "4.13",
    "4.13.lite",
    "4.14",
    "4.14.lite",
    "4.15",
    "4.15.lite",
    "4.16.lite",
    "4.17",
    "4.17.lite",
    "4.18",
]

models_str = ""
for model in models:
    models_str += "'" + model + "', "
models_str = models_str[:-2]



def numpy_to_pil_display(np_array):
    """
    Convert a NumPy array to a PIL Image and display it.
    
    Parameters:
    np_array (numpy.ndarray): The NumPy array to be converted.
    
    Returns:
    PIL.Image.Image: The converted PIL Image.
    """
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(np_array.astype('uint8'))

    # Display the image using ipywidgets
    # img_widget = widgets.Image(
    #     value=pil_image.tobytes(),
    #     format='png',
    #     width="256px"
    # )
    display(pil_image)

    return pil_image

    # Convert and display the image


@torch.inference_mode()
def rife(
    clip: str = "/content/drive/MyDrive/cosmicYT/alchemymagic_100.mp4",
    device_index: int = 0,
    num_streams: int = 2,
    model: str = "4.15",
    use_fp16: bool = False,
    factor_num: int = 6,
    factor_den: int = 2,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = True,
    sc_threshold: float | None = None,
    trt: bool = True,
    trt_debug: bool = False,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [960, 540],
    trt_max_shape: list[int] = [3840, 2160],
    trt_static_shape: bool = True,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> str:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param model:                   Model to use.
    :param factor_num:              Numerator of factor for target frame rate.
    :param factor_den:              Denominator of factor for target frame rate.
                                    For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param fps_num:                 Numerator of target frame rate.
    :param fps_den:                 Denominator of target frame rate.
                                    Override `factor_num` and `factor_den` if specified.
    :param scale:                   Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                                    Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:                Smooth predictions in areas where the estimation is uncertain.
    :param sc:                      Avoid interpolating frames over scene changes.
    :param sc_threshold:            Threshold for scene change detection. Must be between 0.0 and 1.0.
                                    Leave the argument as None if the frames already have _SceneChangeNext property set.
    :param trt:                     Use TensorRT for high-performance inference.
                                    Not supported for '4.0' and '4.1' models.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_min_shape:           Min size of dynamic shapes.
    :param trt_opt_shape:           Opt size of dynamic shapes.
    :param trt_max_shape:           Max size of dynamic shapes.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    print(f'''
    RUNNING WITH THE FOLLOWING PARAMS:
    clip = {clip}
    device_index = {device_index}
    num_streams = {num_streams}
    model = {model}
    use_fp16 = {use_fp16}
    factor_num = {factor_num}
    factor_den = {factor_den}
    fps_num = {fps_num}
    fps_den = {fps_den}
    scale = {scale}
    ensemble = {ensemble}
    sc = {sc}
    sc_threshold = {sc_threshold}
    trt = {trt}
    trt_debug = {trt_debug}
    trt_min_shape = {trt_min_shape}
    trt_opt_shape = {trt_opt_shape}
    trt_max_shape = {trt_max_shape}
    trt_workspace_size = {trt_workspace_size}
    trt_max_aux_streams = {trt_max_aux_streams}
    trt_optimization_level = {trt_optimization_level}
    trt_cache_dir = {trt_cache_dir}
    ''')
    cap = cv2.VideoCapture(clip)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()


    if frame_count < 2:
        raise ValueError("rife: clip's number of frames must be at least 2")

    if not torch.cuda.is_available():
        raise RuntimeError("rife: CUDA is not available")

    if num_streams < 1:
        raise ValueError("rife: num_streams must be at least 1")

    if model not in models:
        raise ValueError(f"rife: model must be {models_str}")

    if factor_num < 1:
        raise ValueError("rife: factor_num must be at least 1")

    if factor_den < 1:
        raise ValueError("rife: factor_den must be at least 1")

    if fps_num is not None and fps_num < 1:
        raise ValueError("rife: fps_num must be at least 1")

    if fps_den is not None and fps_den < 1:
        raise ValueError("rife: fps_den must be at least 1")

    if fps_num is not None and fps_den is not None and fps == 0:
        raise ValueError("rife: clip does not have a valid frame rate and hence fps_num and fps_den cannot be used")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise ValueError("rife: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
        raise TypeError("rife: trt_min_shape must be a list with 2 items")

    if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
        raise TypeError("rife: trt_opt_shape must be a list with 2 items")

    if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
        raise TypeError("rife: trt_max_shape must be a list with 2 items")

    if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
        raise ValueError("rife: trt_min_shape must be less than trt_max_shape")

    if os.path.getsize(os.path.join(model_dir, "flownet_v4.0.pkl")) == 0:
        raise FileNotFoundError("rife: model files have not been downloaded. run 'python -m vsrife' first")

    fp16 = use_fp16
    dtype = torch.half if fp16 else torch.float32

    device = torch.device("cuda", device_index)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(use_fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    torch.set_float32_matmul_precision("high")

    inf_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    f2t_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    t2f_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    inf_stream_locks = [Lock() for _ in range(num_streams)]
    f2t_stream_locks = [Lock() for _ in range(num_streams)]
    t2f_stream_locks = [Lock() for _ in range(num_streams)]

    modulo = 32

    match model:
        case "4.0":
            from .IFNet_HDv3_v4_0 import IFNet
        case "4.1":
            from .IFNet_HDv3_v4_1 import IFNet
        case "4.2":
            from .IFNet_HDv3_v4_2 import IFNet
        case "4.3":
            from .IFNet_HDv3_v4_3 import IFNet
        case "4.4":
            from .IFNet_HDv3_v4_4 import IFNet
        case "4.5":
            from .IFNet_HDv3_v4_5 import IFNet
        case "4.6":
            from .IFNet_HDv3_v4_6 import IFNet
        case "4.7":
            from .IFNet_HDv3_v4_7 import IFNet
        case "4.8":
            from .IFNet_HDv3_v4_8 import IFNet
        case "4.9":
            from .IFNet_HDv3_v4_9 import IFNet
        case "4.10":
            from .IFNet_HDv3_v4_10 import IFNet
        case "4.11":
            from .IFNet_HDv3_v4_11 import IFNet
        case "4.12":
            from .IFNet_HDv3_v4_12 import IFNet
        case "4.12.lite":
            from .IFNet_HDv3_v4_12_lite import IFNet
        case "4.13":
            from .IFNet_HDv3_v4_13 import IFNet
        case "4.13.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet
        case "4.14":
            from .IFNet_HDv3_v4_14 import IFNet
        case "4.14.lite":
            from .IFNet_HDv3_v4_14_lite import IFNet
        case "4.15":
            from .IFNet_HDv3_v4_15 import IFNet
        case "4.15.lite":
            from .IFNet_HDv3_v4_15_lite import IFNet
        case "4.16.lite":
            from .IFNet_HDv3_v4_16_lite import IFNet
        case "4.17":
            from .IFNet_HDv3_v4_17 import IFNet
        case "4.17.lite":
            from .IFNet_HDv3_v4_17_lite import IFNet
        case "4.18":
            from .IFNet_HDv3_v4_18 import IFNet
        case "4.19":
            from .IFNet_HDv3_v4_19 import IFNet
        case "4.20":
            from .IFNet_HDv3_v4_20 import IFNet
        case "4.21":
            from .IFNet_HDv3_v4_21 import IFNet
        case "4.22":
            from .IFNet_HDv3_v4_22 import IFNet
        case "4.22.lite":
            from .IFNet_HDv3_v4_22_lite import IFNet
        case "4.23":
            from .IFNet_HDv3_v4_23 import IFNet
        case "4.24":
            from .IFNet_HDv3_v4_24 import IFNet
        case "4.25":
            from .IFNet_HDv3_v4_25 import IFNet

            modulo = 64
        case "4.25.lite":
            from .IFNet_HDv3_v4_25_lite import IFNet

            modulo = 128
        case "4.26":
            from .IFNet_HDv3_v4_26 import IFNet

            modulo = 64

    model_name = f"flownet_v{model}.pkl"

    # state_dict = torch.load(os.path.join(model_dir, model_name), map_location=device, weights_only=True, mmap=True)
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    # with torch.device("meta"):
    #     flownet = IFNet(scale, ensemble)
    # flownet.load_state_dict(state_dict, strict=False, assign=True)
    # flownet.eval().to(device)
    # if fp16:
    #     flownet.half()

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / fps
        factor_num, factor_den = factor.as_integer_ratio()

    w = width
    h = height
    tmp = max(32, int(32 / scale))
    # pw = math.ceil(w / tmp) * tmp
    # ph = math.ceil(h / tmp) * tmp
    # padding = (0, pw - w, 0, ph - h)
    # tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    tmp = max(modulo, int(modulo / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp
    padding = (0, pw - w, 0, ph - h)

    tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=dtype, device=device)

    tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=dtype, device=device).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
    tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=dtype, device=device).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
    backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    # if sc_threshold is not None:
    #     clip = sc_detect(cv2.VideoCapture(clip), sc_threshold)

    if trt:
        import tensorrt
        import torch_tensorrt

        from .warplayer_custom import WarpPluginCreator

        registry = tensorrt.get_plugin_registry()
        registry.register_creator(WarpPluginCreator())

        if trt_static_shape:
            dimensions = f"{pw}x{ph}"
        else:
            for i in range(2):
                trt_min_shape[i] = math.ceil(trt_min_shape[i] / tmp) * tmp
                trt_opt_shape[i] = math.ceil(trt_opt_shape[i] / tmp) * tmp
                trt_max_shape[i] = math.ceil(trt_max_shape[i] / tmp) * tmp

            dimensions = (
                f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
                f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
                f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
            )

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{dimensions}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_scale-{scale}"
                + f"_ensemble-{ensemble}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")

            flownet = init_module(model_name, IFNet, scale, ensemble, device, dtype)

            example_inputs = (
                torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                torch.zeros([1, 1, ph, pw], dtype=dtype, device=device),
                torch.zeros([2], dtype=torch.float, device=device),
                torch.zeros([1, 2, ph, pw], dtype=torch.float, device=device),
            )

            if trt_static_shape:
                dynamic_shapes = None

                inputs = [
                    torch_tensorrt.Input(shape=[1, 3, ph, pw], dtype=dtype),
                    torch_tensorrt.Input(shape=[1, 3, ph, pw], dtype=dtype),
                    torch_tensorrt.Input(shape=[1, 1, ph, pw], dtype=dtype),
                    torch_tensorrt.Input(shape=[2], dtype=torch.float),
                    torch_tensorrt.Input(shape=[1, 2, ph, pw], dtype=torch.float),
                ]
            else:
                trt_min_shape.reverse()
                trt_opt_shape.reverse()
                trt_max_shape.reverse()

                _height = torch.export.Dim("height", min=trt_min_shape[0] // tmp, max=trt_max_shape[0] // tmp)
                _width = torch.export.Dim("width", min=trt_min_shape[1] // tmp, max=trt_max_shape[1] // tmp)
                dim_height = _height * tmp
                dim_width = _width * tmp
                dynamic_shapes = {
                    "img0": {2: dim_height, 3: dim_width},
                    "img1": {2: dim_height, 3: dim_width},
                    "timestep": {2: dim_height, 3: dim_width},
                    "tenFlow_div": {},
                    "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                }

                inputs = [
                    torch_tensorrt.Input(
                        min_shape=[1, 3] + trt_min_shape,
                        opt_shape=[1, 3] + trt_opt_shape,
                        max_shape=[1, 3] + trt_max_shape,
                        dtype=dtype,
                        name="img0",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 3] + trt_min_shape,
                        opt_shape=[1, 3] + trt_opt_shape,
                        max_shape=[1, 3] + trt_max_shape,
                        dtype=dtype,
                        name="img1",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 1] + trt_min_shape,
                        opt_shape=[1, 1] + trt_opt_shape,
                        max_shape=[1, 1] + trt_max_shape,
                        dtype=dtype,
                        name="timestep",
                    ),
                    torch_tensorrt.Input(
                        shape=[2],
                        dtype=torch.float,
                        name="tenFlow_div",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 2] + trt_min_shape,
                        opt_shape=[1, 2] + trt_opt_shape,
                        max_shape=[1, 2] + trt_max_shape,
                        dtype=torch.float,
                        name="backwarp_tenGrid",
                    ),
                ]

            exported_program = torch.export.export(flownet, example_inputs, dynamic_shapes=dynamic_shapes)

            flownet = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs,
                device=device,
                enabled_precisions={dtype},
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
            )

            # torch_tensorrt.save(flownet, trt_engine_path, output_format="torchscript", inputs=example_inputs)

        flownet = [torch.jit.load(trt_engine_path).eval() for _ in range(num_streams)]
    else:
        flownet = init_module(model_name, IFNet, scale, ensemble, device, dtype)

    warnings.filterwarnings("ignore", "The given NumPy array is not writable")

    index = -1
    index_lock = Lock()

    tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)

    tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
    tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
    tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
    tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
    backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    torch.cuda.current_stream(device).synchronize()
        print('loaded TRT')
    
    def pad_image(img):
        if(use_fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    @torch.inference_mode()
    def inference(n, interp_target, f):
        remainder = interp_target# - n
        # print(f'REMAINDER: {remainder}')

        # if remainder == 0:
        #     # print(f"Frame Type is {type(f[0])} and Frame Shape is {f[0].shape}")
        #     # print('returning f')
        #     return f[0]

        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            # img0 = frame_to_tensor(f[0], device)
            # img1 = frame_to_tensor(f[1], device)
            img0 = f[0]
            img1 = f[1]
            # img0 = img0.permute(0, 3, 1, 2)  # Swap height and width dimensions
            # img1 = img1.permute(0, 3, 1, 2)
            # print(f'img0 shape = {img0.shape}')
            # print(f'img1 shape = {img1.shape}')
            # img0 = F.pad(img0, padding)
            # img1 = F.pad(img1, padding)
            timestep_divisor = (n+1) * 1. / (interp_target+1)
            # print(f'TIMESTEP VALUE {timestep_divisor}')

            timestep = torch.full((1, 1, ph, pw), timestep_divisor, dtype=dtype, device=device)
            # print("img0 shape:", img0.shape)
            # print("img1 shape:", img1.shape)
            # print("timestep shape:", timestep.shape)
            # print("tenFlow_div shape:", tenFlow_div.shape)
            # print("backwarp_tenGrid shape:", backwarp_tenGrid.shape)


            if trt:
                # print('interpolating with trt')
                output = flownet[local_index](img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
            else:
                output = flownet(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)

            torch.cuda.synchronize()

            # print(f"Output Type is {type(output)} and Output Shape is {output.shape}")
            # print('successfully interpolated image')
            # output = output[:, :, :h, :w]
            return output
            # return tensor_to_frame(output[:, :, :h, :w])

    clip_name = os.path.splitext(clip)[0]
    clip_output_path = os.path.join(os.path.dirname(clip), clip_name + "_output_clip.mp4")

    outputfps = int(fps * (factor_num / factor_den))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(clip_output_path, fourcc, outputfps, (width, height))

    videogen = skvideo.io.vreader(clip)
    lastframe = next(videogen)
    h, w, channels = lastframe.shape

    # Initialize the first frame
    # ret, first_frame = cap.read()# Assuming frame_number starts from 0

    use_png = False
    def clear_write_buffer(use_png, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if use_png:
                cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                # print('writing frame', item[:, :, ::-1].shape)
                writer.write(item[:, :, ::-1])

    use_montage = False
    def build_read_buffer(use_montage, read_buffer, videogen):
        try:
            for frame in videogen:
                # if not user_args.img is None:
                #     frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                if use_montage:
                    frame = frame[:, left: left + w]
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (use_png, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (use_montage, write_buffer))

    # writer.write(first_frame)
    # print(f'first frame shape {first_frame.shape}')
    # _, _, channels = first_frame.shape
    # write_buffer.put(first_frame)

    # numpy_to_pil_display(first_frame)

    frame_number = 0

    I1 = torch.from_numpy(np.transpose(np.asarray(lastframe), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)

    temp = None
    times_to_interpolate = int(factor_num / factor_den)
    pbar = tqdm(total=frame_count, desc="Interpolating")

    # while True:
    #     frames = []
    #     start_time = time.time()
    #     if temp is not None:
    #         frame = temp
    #         temp = None
    #     else:
    #         # ret, frame = cap.read()
    #         frame = read_buffer.get()
    #     if frame is None:
    #         break
    #     print(f'FRAME INFORMATION: {frame.shape} TYPE: {type(frame)}')
    #     # frame0 = lastframe
    #     # frame1 = frame
    #     I0 = I1
    #     I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    #     I1 = pad_image(I1)
    #     I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    #     I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    #     # frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    #     ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    #     print(f'Potential SSIM === {ssim}')
    #     ssim = 0.995
    #     break_flag = False
    #     # frames.append(frame0)
    #     # frames.append(I0)

    #     if ssim > 0.996:
    #         # print(f'using ssm greater than 0.996 on frame {frame_number}')
    #         frame = read_buffer.get() # read a new frame
    #         if frame is None:
    #             break_flag = True
    #             frame = lastframe
    #         else:
    #             temp = frame
    #         I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    #         I1 = pad_image(I1)
    #         # print('INTERPOLATING SSIM > 0.996')
    #         # start_time = time.time()
    #         I1 = inference(1, 2, [I0, I1])
    #         # print(f'interpolated frame in {time.time()-start_time:.9f}')
    #         I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    #         ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    #         frame = (I1 * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            
    #     if ssim < 0.2:
    #         # print(f'using ssim < 0.2 on frame {frame_number}')
    #         # output_frames = []
    #         for i in range(times_to_interpolate - 1):
    #             frames.append(I0)
    #         '''
    #         output = []
    #         step = 1 / args.multi
    #         alpha = 0
    #         for i in range(args.multi - 1):
    #             alpha += step
    #             beta = 1-alpha
    #             output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    #         '''
    #     else:
    #         # print(f'not using ssim on frame {frame_number}')
    #         for i in range(times_to_interpolate):
    #             n = times_to_interpolate
    #             # start_time = time.time()
    #             processed_frame = inference(i, n, [I0, I1])
    #             # print(f'interpolated frame in {time.time()-start_time:.9f}')
    #             frames.append(pad_image(processed_frame))
    #             # I1 = I0
    #             # I0 = processed_frame
    #             # if I0.shape != I1.shape:
    #             #     I0 = pad_image(I0)
    #             # if frame0.shape == (width, height, 3):
    #             #     frame0 = frame0.transpose(2, 0, 1)  # Swap dimensions to (3, 540, 960)
    #             #     frame0 = frame0[np.newaxis, :, :, :]
    #             # else:
    #             #     if frame0.shape[0] == 1:
    #             #         frame0 = frame0.squeeze(0)  # Remove the batch dimension only if it's size one

    #             #     frame0 = frame0.transpose(1, 2, 0)
    #             # for j in frames:
    #             #     writer.write(j)
    #             n-=1
    #         # frames.append(frame1)
    #     # frames.append(I1)
        
    #     # writer.write(lastframe)
    #     print(f'writing frame after loop with length = {len(frames)}')
    #     print(f'last frame shape {lastframe.shape}')
    #     write_buffer.put(lastframe)
    #     for j in frames:
    #         j = (((j[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w, :channels]))
    #         print(f'J SHAPE: {j.shape} and TYPE: {type(j)}')
    #         numpy_to_pil_display(j)
    #         write_buffer.put(j)
    start = time.time()

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(np.asarray(frame), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        # print(f'I0 SHAPE: {I0.shape}')
        # print(f'I1 SHAPE: {I1.shape}')
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        # print(f'SSIM ==== {ssim}')

        break_flag = False
        if ssim > 0.996:
            print(f'using ssim > 0.996 on frame : {frame_number}')
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(np.asarray(frame), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = inference(1, 2, [I0, I1])
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            
        if ssim < 0.2:
            print(f'using ssim< 0.2 on frame : {frame_number}')
            output = []
            for i in range(times_to_interpolate - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / args.multi
            alpha = 0
            for i in range(args.multi - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            frames = []
            for i in range(times_to_interpolate-1):
                n = times_to_interpolate-1
                # start_time = time.time()
                processed_frame = inference(i, n, [I0, I1])
                # print(f'interpolated frame in {time.time()-start_time:.9f}')
                frames.append(processed_frame)
                # I1 = I0
                # I0 = processed_frame
                # if I0.shape != I1.shape:
                #     I0 = pad_image(I0)
                # if frame0.shape == (width, height, 3):
                #     frame0 = frame0.transpose(2, 0, 1)  # Swap dimensions to (3, 540, 960)
                #     frame0 = frame0[np.newaxis, :, :, :]
                # else:
                #     if frame0.shape[0] == 1:
                #         frame0 = frame0.squeeze(0)  # Remove the batch dimension only if it's size one
    
                #     frame0 = frame0.transpose(1, 2, 0)
                # for j in frames:
                #     writer.write(j)
                n-=1

        write_buffer.put(lastframe)
        # numpy_to_pil_display(lastframe)
        for mid in frames:
            mid = np.asarray((((mid[0] * 255.).byte().cpu().numpy().astype(np.uint8).transpose(1, 2, 0))))
            # numpy_to_pil_display(mid)
            # print(f'MID SHAPE: {mid.shape} and TYPE: {type(mid)}')
            write_buffer.put(mid[:h, :w])
            # writer.write(j)
        # for mid in processed_frame:
        #     mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        #     write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        frame_number += 1
        # print(f'interpolated loop finished in {time.time()-start_time:.9f}')
        if break_flag:
            break
    # writer.write(lastframe)
        
    # print(f'last frame shape  outside of loop === {lastframe.shape}')
    # numpy_to_pil_display(lastframe)
    write_buffer.put(np.asarray(lastframe))

    while(not write_buffer.empty()):
        time.sleep(0.1)

    
    writer.release()
    # cap.release()
    pbar.close()

    end_time = time.time()
    
    print(f"Time taken to Interpolate: {end_time - start:.2f}")

    return clip_output_path

def init_module(
    model_name: str, IFNet: torch.nn.Module, scale: float, ensemble: bool, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu", weights_only=True, mmap=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    with torch.device("meta"):
        flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval().to(device, dtype)
    return flownet


def sc_detect(frame: np.ndarray, threshold: float) -> np.ndarray:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, sc_mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return sc_mask

def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_normalized = frame.astype(np.uint8) / 255.0
    tensor = torch.from_numpy(frame_normalized).unsqueeze(0).to(device)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    print(f'TENSOR SHAPE: {tensor.shape}')
    frame_normalized = (tensor.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
    return frame_normalized
