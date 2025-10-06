import typing
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import imageio.v2 as imageio
import cv2
from ..types import DatasetFeature, FrozenSet
from ..types import CameraModel, camera_model_to_int, new_cameras
from ..utils import Indices
from ._colmap_utils import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
from ._colmap_utils import read_cameras_text, read_images_text, read_points3D_text, Image, Camera, Point3D
from ._common import DatasetNotFoundError, get_default_viewer_transform, dataset_index_select, new_dataset


def _padded_stack(tensors: Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]) -> np.ndarray:
    if not isinstance(tensors, (tuple, list)):
        return tensors
    max_shape = tuple(max(s) for s in zip(*[x.shape for x in tensors]))
    out_tensors = []
    for x in tensors:
        pads = [(0, m - s) for s, m in zip(x.shape, max_shape)]
        out_tensors.append(np.pad(x, pads))
    return np.stack(out_tensors, 0)


def _parse_colmap_camera_params(camera: Camera) -> Tuple[np.ndarray, int, np.ndarray, Tuple[int, int]]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    out = OrderedDict()  # Default in Python 3.7+
    camera_params = camera.params
    camera_model: CameraModel
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        camera_model = "pinhole"
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        camera_model = "pinhole"
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        camera_model = "opencv"
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        camera_model = "opencv"
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = "opencv"
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = "opencv_fisheye"
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        camera_model = "opencv_fisheye"
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = "opencv_fisheye"
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    image_width: int = camera.width
    image_height: int = camera.height
    intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32)
    distortion_params = np.array([out.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4")], dtype=np.float32)
    return intrinsics, camera_model_to_int(camera_model), distortion_params, (image_width, image_height)


def load_colmap_dataset(path: Union[Path, str],
        images_path: Optional[Union[str, Path]] = None, 
        split: Optional[str] = None, 
        test_indices: Optional[Indices] = None,
        features: Optional[FrozenSet[DatasetFeature]] = None,
        colmap_path: Optional[Union[str, Path]] = None):
    path = Path(path)
    colmap_path = Path(colmap_path) if colmap_path is not None else None
    if features is None:
        features = typing.cast(FrozenSet[DatasetFeature], {})
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}
    # Load COLMAP dataset
    if colmap_path is None:
        colmap_path = Path("sparse") / "0"
        if not (path / colmap_path).exists():
            colmap_path = Path("sparse")
    colmap_path = path / colmap_path
    if images_path is None:
        images_path = Path("images_8")
    images_path = path / images_path
    if not colmap_path.exists():
        raise DatasetNotFoundError("Missing 'sparse/0' folder in COLMAP dataset")
    if not (colmap_path / "cameras.bin").exists() and not (colmap_path / "cameras.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")
    if not images_path.exists():
        raise DatasetNotFoundError("Missing 'images' folder in COLMAP dataset")

    if (colmap_path / "cameras.bin").exists():
        colmap_cameras = read_cameras_binary(colmap_path / "cameras.bin")
    elif (colmap_path / "cameras.txt").exists():
        colmap_cameras = read_cameras_text(colmap_path / "cameras.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")

    if not (colmap_path / "images.bin").exists() and not (colmap_path / "images.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")
    if (colmap_path / "images.bin").exists():
        images = read_images_binary(colmap_path / "images.bin")
    elif (colmap_path / "images.txt").exists():
        images = read_images_text(colmap_path / "images.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")

    points3D: Optional[Dict[int, Point3D]] = None
    if load_points:
        if not (colmap_path / "points3D.bin").exists() and not (colmap_path / "points3D.txt").exists():
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")
        if (colmap_path / "points3D.bin").exists():
            points3D = read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            points3D = read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")

    # Convert to tensors
    camera_intrinsics = []
    camera_poses = []
    camera_models = []
    camera_distortion_params = []
    image_paths: List[str] = []
    image_names = []
    camera_sizes = []

    # sorted
    images = dict(sorted(images.items(), key=lambda kv: kv[1].name))


    image: Image

    # --- Step 0: Check and compute rescale factors from actual images ---
    sample_image_path = next((path / "images_8").glob("*"))  # pick the first image in folder
    sample_image = imageio.imread(sample_image_path)[..., :3]
    actual_height, actual_width = sample_image.shape[:2]
    
    # colmap run on images
    sample_image_path = next((path / "images").glob("*"))  # pick the first image in folder
    sample_image = imageio.imread(sample_image_path)[..., :3]
    colmap_height, colmap_width = sample_image.shape[:2]

    # Assume all images have same size (typical COLMAP dataset)
    s_width = actual_width / colmap_width
    s_height = actual_height / colmap_height


    # --- Step 1: Pre-loop to collect ROI + new_K info ---
    all_rois = []
    all_newKs = []
    all_maps = []
    image_sizes = []
    for image in images.values():
        camera: Camera = colmap_cameras[image.camera_id]
        intrinsics, camera_model, distortion_params, (w, h) = _parse_colmap_camera_params(camera)

        # intrinsics is 1-D; scale elements directly  
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.size == 4:          # PINHOLE: [fx, fy, cx, cy]   
            fx, fy, cx, cy = intrinsics
        else:
            raise ValueError(f"Unsupported intrinsics length: {intrinsics.size}")

        fx *= s_width                     
        fy *= s_height                    
        cx *= s_width                     
        cy *= s_height                    

        w = int(round(w * s_width))       
        h = int(round(h * s_height))      
        assert w == actual_width
        assert h == actual_height         

        # Build 3x3 K for OpenCV from the scaled 1-D intrinsics
        K = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

        D = np.asarray(distortion_params[:4], dtype=np.float32)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

        # Precompute undistort maps (store as fixed-point to save memory)
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=K, distCoeffs=D, R=np.eye(3, dtype=np.float32),
            newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_16SC2
        )

        all_rois.append(roi)
        all_newKs.append(new_K)
        all_maps.append((map1, map2))
        image_sizes.append((w, h))

    # --- Compute global crop bounding box (unchanged) ---
    x_min = max([r[0] for r in all_rois])
    y_min = max([r[1] for r in all_rois])
    x_max = min([r[0] + r[2] for r in all_rois])
    y_max = min([r[1] + r[3] for r in all_rois])

    
    global_crop = (x_min, y_min, x_max, y_max)
    crop_w = x_max - x_min
    crop_h = y_max - y_min

    i = 0
    c2w: np.ndarray
    
    undistort_map_dict = {} 

    for image, new_K in zip(images.values(), all_newKs):
        camera: Camera = colmap_cameras[image.camera_id]
        intrinsics, camera_model, distortion_params, (w, h) = _parse_colmap_camera_params(camera)

        # adjust K
        new_K_adj = new_K.copy()
        new_K_adj[0, 2] -= x_min  # shift principal point
        new_K_adj[1, 2] -= y_min  # shift principal point

        fx = new_K_adj[0, 0]
        fy = new_K_adj[1, 1]
        cx = new_K_adj[0, 2]
        cy = new_K_adj[1, 2]
        intrinsic_vec = np.array([fx, fy, cx, cy], dtype=np.float32)
        camera_intrinsics.append(intrinsic_vec)
        camera_models.append(0) # PINHOLE
        camera_distortion_params.append(np.zeros(0, dtype=np.float32))
        camera_sizes.append(np.array((crop_w, crop_h), dtype=np.int32))  # from (w, h) → (crop_w, crop_h)
        image_names.append(image.name)
        image_paths.append(str(images_path / image.name))

        # Store map & crop info for later image loading
        undistort_map_dict[str(images_path / image.name)] = {
            "map1": map1,
            "map2": map2,
            "crop": (x_min, y_min, x_max, y_max)
        }

        rotation = qvec2rotmat(image.qvec).astype(np.float64)
        translation = image.tvec.reshape(3, 1).astype(np.float64)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=w2c.dtype)], 0)
        c2w = np.linalg.inv(w2c)
        camera_poses.append(c2w[0:3, :])

        i += 1

    #for image in images.values():
    #    camera: Camera = colmap_cameras[image.camera_id]
    #    intrinsics, camera_model, distortion_params, (w, h) = _parse_colmap_camera_params(camera)
    #    camera_sizes.append(np.array((w, h), dtype=np.int32))
    #    camera_intrinsics.append(intrinsics)
    #    camera_models.append(camera_model)
    #    camera_distortion_params.append(distortion_params)
    #    image_names.append(image.name)
    #    image_paths.append(str(images_path / image.name))
    #
    #    rotation = qvec2rotmat(image.qvec).astype(np.float64)

    #    translation = image.tvec.reshape(3, 1).astype(np.float64)
    #    w2c = np.concatenate([rotation, translation], 1)
    #    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=w2c.dtype)], 0)
    #    c2w = np.linalg.inv(w2c)

    #    camera_poses.append(c2w[0:3, :])
    #    i += 1

    # Estimate nears fars
    near = 0.01
    far = np.stack([x[:3, -1] for x in camera_poses], 0)
    far = float(np.percentile(np.linalg.norm(far - np.mean(far, keepdims=True, axis=0), axis=-1), 90, axis=0))
    nears_fars = np.array([[near, far]] * len(camera_poses), dtype=np.float32)

    # Load points
    points3D_xyz = None
    points3D_rgb = None
    if load_points:
        assert points3D is not None, "3D points have not been loaded"
        points3D_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points3D_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

    # camera_ids=torch.tensor(camera_ids, dtype=torch.int32),
    all_cameras = new_cameras(
        poses=np.stack(camera_poses, 0).astype(np.float32),
        intrinsics=np.stack(camera_intrinsics, 0).astype(np.float32),
        camera_models=np.array(camera_models, dtype=np.int32),
        distortion_parameters=_padded_stack(camera_distortion_params).astype(np.float32),
        image_sizes=np.stack(camera_sizes, 0).astype(np.int32),
        nears_fars=nears_fars.astype(np.float32),
    )
    indices = None
    train_indices = np.arange(len(image_paths))
    if split is not None:
        if test_indices is None and ((path / "train_list.txt").exists() or (path / "test_list.txt").exists()):
            logging.info(f"colmap dataloader is loading split data from {path / f'{split}_list.txt'}")
            train_indices = None
            for split in ("train", split):
                split_image_names = set((path / f"{split}_list.txt").read_text().splitlines())
                indices = np.array([name in split_image_names for i, name in enumerate(image_names)], dtype=bool)
                if indices.sum() == 0:
                    raise DatasetNotFoundError(f"no images found for split {split} in {path / f'{split}_list.txt'}")
                if indices.sum() < len(split_image_names):
                    logging.warning(f"only {indices.sum()} images found for split {split} in {path / f'{split}_list.txt'}")
                if split == "train":
                    train_indices = indices
            assert train_indices is not None
        else:
            # TODO: clutter here
            #if test_indices is None:
            #    test_indices = Indices.every_iters(8)
            #dataset_len = len(image_paths)
            #test_indices.total = dataset_len
            #test_indices_array: np.ndarray = np.array([i in test_indices for i in range(dataset_len)], dtype=bool)
            #train_indices = np.logical_not(test_indices_array)
            #indices = train_indices if split == "train" else test_indices_array
            
            # split training and testing
            image_names_np = np.array([p.split("/")[-1] for p in image_paths])
            train_mask = np.array(["clutter_" in name for name in image_names_np], dtype=bool)
            test_mask = np.array(["extra_" in name for name in image_names_np], dtype=bool)
            if split == "train":
                indices = train_mask
            elif split == "test":
                indices = test_mask
            else:
                raise ValueError(f"Unknown split: {split}")

    viewer_transform, viewer_pose = get_default_viewer_transform(all_cameras[train_indices].poses, None)
    dataset = new_dataset(
        cameras=all_cameras,
        image_paths=image_paths,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        mask_paths=None,
        image_paths_root=str(images_path),
        metadata={
            "name": None,
            "color_space": "srgb",
            "viewer_transform": viewer_transform,
            "viewer_initial_pose": viewer_pose,
        },
        undistort_map=undistort_map_dict,
        )
    if indices is not None:
        dataset = dataset_index_select(dataset, indices)
    #print(dataset)

    return dataset
