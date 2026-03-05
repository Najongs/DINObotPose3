"""
DINOv3 Pose Estimation Dataset
DREAM 데이터셋 구조를 따르는 데이터로더
"""

import os
import json
import random
import glob
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as albu
from typing import Dict, List, Tuple, Optional


def fda_transfer(src_img: np.ndarray, trg_img: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    FDA (Fourier Domain Adaptation): Replace low-frequency spectrum of source with target's.
    Low-freq = overall color/tone (domain-specific), High-freq = edges/structure (task-relevant).

    Args:
        src_img: Synthetic image (H, W, 3), uint8
        trg_img: Real image (H, W, 3), uint8
        beta: Low-frequency replacement ratio (0.01~0.05 recommended for DR data)
    Returns:
        FDA-applied image (H, W, 3), uint8
    """
    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)

    # Resize target to match source
    if src.shape[:2] != trg.shape[:2]:
        trg = np.array(PILImage.fromarray(trg.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0]), PILImage.BILINEAR
        )).astype(np.float32)

    result = np.zeros_like(src)
    h, w = src.shape[:2]
    cy, cx = h // 2, w // 2
    bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)

    for ch in range(3):
        fft_src = np.fft.fftshift(np.fft.fft2(src[:, :, ch]))
        fft_trg = np.fft.fftshift(np.fft.fft2(trg[:, :, ch]))

        amp_src = np.abs(fft_src)
        phase_src = np.angle(fft_src)
        amp_trg = np.abs(fft_trg)

        # Replace low-frequency amplitude
        amp_src[cy - bh:cy + bh, cx - bw:cx + bw] = amp_trg[cy - bh:cy + bh, cx - bw:cx + bw]

        fft_result = np.fft.ifftshift(amp_src * np.exp(1j * phase_src))
        result[:, :, ch] = np.real(np.fft.ifft2(fft_result))

    return np.clip(result, 0, 255).astype(np.uint8)


# Robot type constants (must match model.py)
ROBOT_TYPE_NAMES = ['franka_panda', 'meca500', 'fr5', 'franka_research3']


def infer_robot_type_from_path(path: str) -> int:
    """
    Infer robot type from directory path based on naming conventions.

    Args:
        path: Directory path containing the dataset

    Returns:
        Robot type index (0-3) corresponding to ROBOT_TYPE_NAMES
    """
    path_lower = path.lower()

    # Check for specific patterns (order matters - more specific first)
    if 'research3' in path_lower:
        return 3  # franka_research3
    elif 'Fr5' in path_lower:
        return 2  # fr5
    elif 'Meca' in path_lower:
        return 1  # meca500
    elif 'panda' in path_lower or 'dream' in path_lower:
        return 0  # franka_panda
    else:
        # Default to franka_panda if no match
        return 0


class PoseEstimationDataset(Dataset):
    """
    DREAM 스타일의 NDDS 데이터셋을 위한 데이터로더

    데이터 구조:
    - RGB 이미지
    - JSON 어노테이션 (keypoint 위치)
    - (선택적) Joint angle 정보
    """

    def __init__(
        self,
        data_dir: str,
        keypoint_names: List[str],
        image_size: Tuple[int, int] = (512, 512),
        heatmap_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        normalize: bool = True,
        include_angles: bool = True,
        sigma: float = 5.0,  # Gaussian heatmap sigma
        multi_robot: bool = False,  # Load data from multiple robot subdirectories
        robot_types: Optional[List[str]] = None,  # List of robot types to include
        fda_real_dir: Optional[str] = None,  # Real image directory for FDA augmentation
        fda_beta: float = 0.01,  # FDA low-frequency replacement ratio
        fda_prob: float = 0.5,  # Probability of applying FDA per sample
        occlusion_prob: float = 0.0,  # Probability of synthetic occlusion augmentation
        occlusion_max_holes: int = 6,  # Max number of coarse occlusion patches
        occlusion_max_size_frac: float = 0.2,  # Max occluder size relative to image side
        json_allowlist_path: Optional[str] = None,  # Optional list file (txt/json) to keep only selected json frames
    ):
        """
        Args:
            data_dir: NDDS 데이터가 있는 디렉토리
            keypoint_names: 키포인트 이름 리스트 (예: ['panda_link0', ...])
            image_size: 네트워크 입력 이미지 크기
            heatmap_size: 출력 heatmap 크기
            augment: 데이터 증강 사용 여부
            normalize: 이미지 정규화 여부
            include_angles: joint angle 정보 포함 여부
            sigma: Gaussian heatmap의 표준편차
            multi_robot: True면 data_dir 하위의 모든 로봇 데이터를 통합하여 로드
            robot_types: multi_robot=True일 때 특정 로봇 타입만 필터링 (예: ['panda', 'kuka'])
            fda_real_dir: Real 이미지 디렉토리 (FDA style source, label 불필요)
            fda_beta: FDA 저주파 교체 비율 (0.01=미세한 톤 변화, 0.05=강한 변환)
            fda_prob: FDA 적용 확률 (0.5 = 50%의 샘플에 적용)
            occlusion_prob: 가려짐 증강(CoarseDropout) 적용 확률
            occlusion_max_holes: 최대 가림 패치 개수
            occlusion_max_size_frac: 가림 패치 최대 크기 비율(이미지 변 길이 대비)
            json_allowlist_path: 선택된 frame json 이름/경로 리스트 파일(txt/json)
        """
        self.data_dir = data_dir
        self.keypoint_names = keypoint_names
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment
        self.include_angles = include_angles
        self.sigma = sigma
        self.multi_robot = multi_robot
        self.robot_types = robot_types
        self.fda_beta = fda_beta
        self.fda_prob = fda_prob
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_holes = max(1, int(occlusion_max_holes))
        self.occlusion_max_size_frac = max(0.01, float(occlusion_max_size_frac))
        self.json_allowlist_path = json_allowlist_path
        self.json_allowlist_keys = self._load_json_allowlist(json_allowlist_path)

        # FDA: Load real image paths for style transfer
        self.fda_real_paths = []
        if fda_real_dir and os.path.isdir(fda_real_dir):
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                self.fda_real_paths.extend(glob.glob(os.path.join(fda_real_dir, '**', ext), recursive=True))
            if self.fda_real_paths:
                print(f"FDA enabled: {len(self.fda_real_paths)} real images from {fda_real_dir} (beta={fda_beta}, prob={fda_prob})")
            else:
                print(f"FDA warning: No images found in {fda_real_dir}")

        # 데이터 파일 리스트 로드
        self.samples = self._load_dataset()

        # 이미지 변환 설정
        if normalize:
            # ImageNet 통계값 사용 (DINOv3가 학습된 방식)
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])

        # 데이터 증강 설정
        if self.augment:
            # 🚀 [적당한 수준] Sim-to-Real 일반화 + 수렴 우선
            occ_max_frac = max(0.05, min(0.6, float(self.occlusion_max_size_frac)))
            occ_min_frac = min(0.04, occ_max_frac)
            self.augmentation = albu.Compose([
                # 1. 적당한 노이즈
                albu.GaussNoise(p=0.3),

                # 2. 적당한 색상 변화
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                albu.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),

                # 3. Occlusion (적당한 수준)
                albu.CoarseDropout(
                    num_holes_range=(1, max(3, self.occlusion_max_holes)),
                    hole_height_range=(0.08, 0.3),
                    hole_width_range=(0.08, 0.3),
                    fill=0,
                    p=0.4,
                ),

                # 4. 기하학적 변환 (제한적)
                albu.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.3
                ),

            ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))

    @staticmethod
    def _normalize_path_token(path_str: str) -> str:
        return os.path.normpath(path_str).replace('\\', '/').lower()

    def _load_json_allowlist(self, allowlist_path: Optional[str]) -> Optional[set]:
        if not allowlist_path:
            return None

        if not os.path.exists(allowlist_path):
            raise FileNotFoundError(f"json allowlist not found: {allowlist_path}")

        keys = set()
        suffix = Path(allowlist_path).suffix.lower()

        def add_key(token: str):
            token = str(token).strip()
            if not token:
                return
            keys.add(self._normalize_path_token(token))
            name = os.path.basename(token)
            if name:
                keys.add(name.lower())
                stem = os.path.splitext(name)[0]
                if stem:
                    keys.add(stem.lower())

        if suffix == '.json':
            with open(allowlist_path, 'r') as f:
                payload = json.load(f)

            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        add_key(item)
                    elif isinstance(item, dict):
                        for field in ('json_path', 'json_name', 'name'):
                            if field in item and item[field]:
                                add_key(item[field])
            elif isinstance(payload, dict):
                for field in ('json_paths', 'json_names', 'items'):
                    val = payload.get(field)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                add_key(item)
                            elif isinstance(item, dict):
                                for f2 in ('json_path', 'json_name', 'name'):
                                    if f2 in item and item[f2]:
                                        add_key(item[f2])
            else:
                raise ValueError(f"Unsupported allowlist JSON format: {allowlist_path}")
        else:
            with open(allowlist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    add_key(line)

        if not keys:
            raise ValueError(f"json allowlist is empty: {allowlist_path}")

        print(f"JSON allowlist loaded: {allowlist_path} ({len(keys)} match keys)")
        return keys

    def _is_json_allowed(self, json_path: str, json_file: str) -> bool:
        if self.json_allowlist_keys is None:
            return True

        p = Path(json_path)
        candidates = {
            self._normalize_path_token(json_path),
            self._normalize_path_token(str(p.resolve())),
            json_file.lower(),
            p.stem.lower(),
        }
        try:
            rel = p.resolve().relative_to(Path(self.data_dir).resolve())
            candidates.add(self._normalize_path_token(str(rel)))
        except Exception:
            pass
        return any(c in self.json_allowlist_keys for c in candidates)

    def _load_dataset(self) -> List[Dict]:
        """NDDS 데이터셋에서 샘플 리스트 로드"""
        samples = []

        if self.multi_robot:
            # Multi-robot mode: load from all subdirectories
            print(f"Loading multi-robot dataset from {self.data_dir}")
            robot_dirs = []

            # DREAM data structure: /data/real/* and /data/synthetic/*
            # Recursively find all robot-specific directories
            def find_robot_dirs(base_path, max_depth=3, current_depth=0):
                """Recursively find directories containing robot data"""
                found_dirs = []
                if current_depth > max_depth:
                    return found_dirs

                try:
                    items = os.listdir(base_path)
                except (PermissionError, FileNotFoundError) as e:
                    print(f"  Warning: Cannot access {base_path}: {e}")
                    return found_dirs

                for item in items:
                    item_path = os.path.join(base_path, item)
                    if not os.path.isdir(item_path):
                        continue

                    # Check if this directory contains data files (has .json files)
                    try:
                        dir_files = os.listdir(item_path)
                        has_json = any(f.endswith('.json') for f in dir_files)
                    except (PermissionError, FileNotFoundError):
                        has_json = False

                    if has_json:
                        # This is a data directory, check if it matches robot filter
                        if self.robot_types:
                            if any(robot_type.lower() in item.lower() for robot_type in self.robot_types):
                                print(f"  Adding data directory: {item_path}")
                                found_dirs.append(item_path)
                        else:
                            print(f"  Adding data directory: {item_path}")
                            found_dirs.append(item_path)
                    else:
                        # Recurse into subdirectories
                        found_dirs.extend(find_robot_dirs(item_path, max_depth, current_depth + 1))

                return found_dirs

            print(f"Searching for robot type(s): {self.robot_types}")
            robot_dirs = find_robot_dirs(self.data_dir)

            print(f"Found {len(robot_dirs)} robot data directories")
            for rdir in robot_dirs:
                print(f"  - {rdir}")
                samples.extend(self._load_from_directory(rdir))
        else:
            # Single directory mode
            samples = self._load_from_directory(self.data_dir)

        print(f"Loaded {len(samples)} samples total")
        return samples

    def _load_from_directory(self, directory: str) -> List[Dict]:
        """단일 디렉토리에서 샘플 로드"""
        samples = []

        # Infer robot type from directory path
        robot_type = infer_robot_type_from_path(directory)

        # NDDS 형식: 각 프레임마다 이미지와 .json 파일이 쌍으로 존재
        for root, dirs, files in os.walk(directory):
            json_files = [f for f in files if f.endswith('.json') and not f.startswith('_')]

            for json_file in json_files:
                json_path = os.path.join(root, json_file)
                if not self._is_json_allowed(json_path, json_file):
                    continue
                base_name = json_file.replace('.json', '')
                img_path = None

                # Try to read image path from JSON meta
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'meta' in data and 'image_path' in data['meta']:
                            # Get image path from JSON (can be relative or absolute)
                            json_img_path = data['meta']['image_path']

                            # Fix incorrect relative path: ../dataset/... should be ../../../...
                            if json_img_path.startswith('../dataset/'):
                                json_img_path = json_img_path.replace('../dataset/', '../../../', 1)

                            # If relative path, resolve from JSON directory
                            if not os.path.isabs(json_img_path):
                                img_path = os.path.normpath(os.path.join(root, json_img_path))
                            else:
                                img_path = json_img_path

                            # Verify image exists
                            if not os.path.exists(img_path):
                                print(f"Warning: Image not found: {img_path}")
                                img_path = None
                except Exception as e:
                    print(f"Warning: Failed to read {json_path}: {e}")
                    img_path = None

                # Fallback: look for image in same directory as JSON
                if img_path is None:
                    for ext in ['.rgb.jpg', '.png', '.jpg', '.jpeg']:
                        potential_path = os.path.join(root, base_name + ext)
                        if os.path.exists(potential_path):
                            img_path = potential_path
                            break

                if img_path:
                    # Detect synthetic data (DREAM sim uses cm, real uses m)
                    is_synthetic = 'syn' in directory.lower()
                    samples.append({
                        'image_path': img_path,
                        'annotation_path': json_path,
                        'name': base_name,
                        'source_dir': os.path.basename(directory),
                        'robot_type': robot_type,
                        'is_synthetic': is_synthetic
                    })

        return samples

    def _load_keypoints_from_json(self, json_path: str) -> Dict:
        """JSON 파일에서 keypoint 정보 로드"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        keypoints = {}

        # Initialize keypoint positions array with the correct order
        # This ensures keypoints are in self.keypoint_names order, not JSON order
        keypoint_positions = np.zeros((len(self.keypoint_names), 2), dtype=np.float32)
        keypoint_positions_3d = np.zeros((len(self.keypoint_names), 3), dtype=np.float32)
        keypoint_found = [False] * len(self.keypoint_names)

        # NDDS 형식에서 keypoint 추출
        if 'objects' in data:
            for obj in data['objects']:
                if 'keypoints' in obj:
                    for kp in obj['keypoints']:
                        kp_name = kp['name']
                        # 부분 일치 검사 (예: 'panda_link0'에서 'link0' 찾기)
                        target_idx = -1
                        for i, name in enumerate(self.keypoint_names):
                            if name in kp_name:
                                target_idx = i
                                break
                        
                        if target_idx != -1:
                            keypoint_positions[target_idx] = [
                                kp['projected_location'][0],
                                kp['projected_location'][1]
                            ]
                            if 'location' in kp:
                                keypoint_positions_3d[target_idx] = [
                                    kp['location'][0],
                                    kp['location'][1],
                                    kp['location'][2]
                                ]
                            keypoint_found[target_idx] = True

        # Mark missing keypoints with negative coordinates
        for i, found in enumerate(keypoint_found):
            if not found:
                keypoint_positions[i] = [-1, -1]
                keypoint_positions_3d[i] = [-1, -1, -1]

        keypoints['projections'] = keypoint_positions
        keypoints['locations'] = keypoint_positions_3d

        # Camera intrinsic matrix K (from meta.K)
        if 'meta' in data and 'K' in data['meta']:
            keypoints['camera_K'] = np.array(data['meta']['K'], dtype=np.float32)
        else:
            # Default fallback (should not happen with proper data)
            keypoints['camera_K'] = np.eye(3, dtype=np.float32)

        # Joint angles from sim_state.joints (first 7 joint positions)
        if self.include_angles and 'sim_state' in data and 'joints' in data['sim_state']:
            joints = data['sim_state']['joints']
            angles = np.array([j['position'] for j in joints[:7]], dtype=np.float32)
            keypoints['angles'] = angles  # radians, no normalization (FK needs raw radians)

        return keypoints

    def _create_heatmap(self, keypoints: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Keypoint 위치로부터 Gaussian heatmap 생성 (최적화 버전)
        """
        H, W = size
        num_keypoints = len(keypoints)
        heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

        # meshgrid를 한 번만 생성
        x_range = np.arange(W)
        y_range = np.arange(H)
        xx, yy = np.meshgrid(x_range, y_range)

        for i, (x, y) in enumerate(keypoints):
            # 이미지 범위를 벗어난 키포인트 처리 (Albumentations 이후 대비)
            if x < 0 or y < 0 or x >= W or y >= H:
                continue

            # Gaussian 생성
            # sigma가 너무 작으면 학습이 안 되므로 최소값 보장
            sigma = max(self.sigma, 1.0)
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            heatmap = np.exp(-d2 / (2 * sigma ** 2))
            
            # 아주 작은 값은 0으로 처리하여 sparsity 확보
            heatmap[heatmap < 0.01] = 0
            heatmaps[i] = heatmap

        return heatmaps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # 이미지 로드
        image = PILImage.open(sample_info['image_path']).convert('RGB')
        original_size = image.size  # (W, H)

        # Keypoint 로드
        keypoints_data = self._load_keypoints_from_json(sample_info['annotation_path'])
        keypoints = keypoints_data['projections'].copy()  # (N, 2) [x, y]

        # FDA augmentation (applied before other augmentations)
        if self.fda_real_paths and random.random() < self.fda_prob:
            real_path = random.choice(self.fda_real_paths)
            try:
                real_img = np.array(PILImage.open(real_path).convert('RGB'))
                src_img = np.array(image)
                image = PILImage.fromarray(fda_transfer(src_img, real_img, beta=self.fda_beta))
            except Exception:
                pass  # Skip FDA on error, use original image

        # 데이터 증강 적용
        if self.augment and len(keypoints) > 0:
            augmented = self.augmentation(
                image=np.array(image),
                keypoints=keypoints
            )
            image = PILImage.fromarray(augmented['image'])
            keypoints = np.array(augmented['keypoints'])

        # 이미지 크기 변경에 따른 keypoint 좌표 조정
        scale_x = self.heatmap_size[1] / original_size[0]
        scale_y = self.heatmap_size[0] / original_size[1]
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] *= scale_x
        keypoints_scaled[:, 1] *= scale_y

        # 이미지 변환
        image_tensor = self.transform(image)

        # Heatmap 생성
        heatmaps = self._create_heatmap(keypoints_scaled, self.heatmap_size)
        heatmaps_tensor = torch.from_numpy(heatmaps).float()

        # Keypoint 좌표를 텐서로
        keypoints_tensor = torch.from_numpy(keypoints_scaled).float()
        keypoints_3d_tensor = torch.from_numpy(keypoints_data['locations']).float()

        # Synthetic data (DREAM sim) uses cm, convert to meters
        if sample_info.get('is_synthetic', False):
            keypoints_3d_tensor = keypoints_3d_tensor / 100.0

        # Create valid mask (True for keypoints with valid coordinates)
        valid_mask = torch.tensor([kp[0] >= 0 and kp[1] >= 0 for kp in keypoints_scaled], dtype=torch.bool)

        # Get robot type from sample info
        robot_type = sample_info.get('robot_type', 0)  # Default to franka_panda if not found

        # Camera intrinsic matrix (original resolution, will be scaled in train_3d.py)
        camera_K = torch.from_numpy(keypoints_data['camera_K']).float()  # (3, 3) - original resolution

        # 🚀 [NEW] Extract depths from 3D keypoints (Z-coordinate in camera frame)
        depths = keypoints_3d_tensor[:, 2]  # (num_joints,) - Z values in meters

        sample = {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'keypoints': keypoints_tensor,
            'keypoints_3d': keypoints_3d_tensor,
            'depths': depths,  # 🚀 [NEW] Depth ground truth
            'valid_mask': valid_mask,
            'robot_type': torch.tensor(robot_type, dtype=torch.long),
            'name': sample_info['name'],
            'annotation_path': sample_info['annotation_path'],
            'camera_K': camera_K,
            'original_size': torch.tensor([original_size[0], original_size[1]], dtype=torch.float32),  # (W, H)
        }

        # Joint angles 포함 (있는 경우)
        if self.include_angles and 'angles' in keypoints_data:
            angles = keypoints_data['angles']
            sample['angles'] = torch.from_numpy(angles).float()
            sample['has_angles'] = torch.tensor(True, dtype=torch.bool)
        else:
            # Dummy angles (모델이 angle 출력을 요구하는 경우)
            sample['angles'] = torch.zeros(7).float()  # 7 joint angles for Panda
            sample['has_angles'] = torch.tensor(False, dtype=torch.bool)

        return sample


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    keypoint_names: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (512, 512),
    val_split: float = 1.0,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Train/Validation 데이터로더 생성

    Args:
        train_dir: 학습 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        keypoint_names: 키포인트 이름 리스트
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        image_size: 입력 이미지 크기
        heatmap_size: 출력 heatmap 크기
        val_split: 검증 데이터 사용 비율 (0.0~1.0, default=1.0 for all data)

    Returns:
        train_loader, val_loader
    """
    train_dataset = PoseEstimationDataset(
        data_dir=train_dir,
        keypoint_names=keypoint_names,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=True,
        **kwargs
    )

    val_dataset_full = PoseEstimationDataset(
        data_dir=val_dir,
        keypoint_names=keypoint_names,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False,
        **kwargs
    )

    # Use only a fraction of validation data if val_split < 1.0
    if val_split < 1.0:
        val_size = int(len(val_dataset_full) * val_split)
        unused_size = len(val_dataset_full) - val_size
        generator = torch.Generator().manual_seed(42)
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset_full, [val_size, unused_size], generator=generator
        )
    else:
        val_dataset = val_dataset_full

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 테스트 코드
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    dataset = PoseEstimationDataset(
        data_dir="/path/to/your/data",
        keypoint_names=keypoint_names,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Heatmaps shape: {sample['heatmaps'].shape}")
        print(f"Keypoints shape: {sample['keypoints'].shape}")
        if 'angles' in sample:
            print(f"Angles shape: {sample['angles'].shape}")
