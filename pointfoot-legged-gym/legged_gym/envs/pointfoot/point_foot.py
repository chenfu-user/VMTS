#encoding=utf8
import os
from typing import Dict

import torch, torchvision
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain
import open3d as o3d
import numpy as np
import cv2
import warp as wp
import warp.render as wprender
import time


@wp.kernel
def depth_draw(mesh: wp.uint64, cam_pos: wp.vec3, cam_rot: wp.quat, width: int, height: int, fov: float, pixels: wp.array(dtype=float)):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    y = height - y - 1  # 将y坐标反转

    fov_rad = np.radians(fov)  # 将FOV转换为弧度
    scale = np.tan(fov_rad / 2.0)  # 根据 FOV 计算 scale

    # 计算视图中的坐标，考虑宽高比（aspect ratio）
    aspect_ratio = float(width) / float(height)

    sx = 2.0 * float(x) / float(width) - 1.0  # 横向投射
    sy = 2.0 * float(y) / float(height) - 1.0  # 纵向投射
    view_direction = wp.vec3(sx*scale, sy*scale, -1.0 * aspect_ratio)
    init_quat = wp.quat_rpy(np.pi/2.0, 0.0, -np.pi/2.0)
    rd = wp.quat_rotate(init_quat, wp.normalize(view_direction))

    # 将旋转应用到视线方向
    rd = wp.quat_rotate(cam_rot, rd)  # 应用相机的旋转
    # rd = wp.normalize(view_direction)
    # 计算视线的起点和方向
    offset = wp.vec3(25.0, 25.0, 0.0)  # 设定偏移量
    ro = cam_pos + offset  # 在外部完成加法操作
    # rd = wp.normalize(rd)

    # 光线投射查询
    query = wp.mesh_query_ray(mesh, ro, rd, 3.0)

    # 将查询结果赋值到像素数组中
    pixels[tid] = query.t


class IsaacGymImuFilter:
    def __init__(self, num_envs, device, alpha=0.7):
        """
        初始化IMU滤波器
        Args:
            num_envs: 环境数量
            device: 计算设备 (cuda/cpu)
            alpha: 低通滤波系数 (0-1)
        """
        self.num_envs = num_envs
        self.device = device
        self.alpha = alpha
        
        # 初始化上一次的状态值
        self.last_base_ang_vel = torch.zeros((num_envs, 3), device=device)
        self.last_base_quat = torch.zeros((num_envs, 4), device=device)
        self.last_projected_gravity = torch.zeros((num_envs, 3), device=device)
        
    def filter_ang_vel(self, current_ang_vel):
        """
        对角速度进行低通滤波
        """
        filtered_ang_vel = self.alpha * current_ang_vel + (1.0 - self.alpha) * self.last_base_ang_vel
        self.last_base_ang_vel = filtered_ang_vel
        return filtered_ang_vel
    
    def filter_orientation(self, base_quat, gravity_vec=torch.tensor([0., 0., -1.])):
        """
        处理方向四元数和重力投影
        """
        # 确保重力向量在正确的设备上
        if gravity_vec.device != self.device:
            gravity_vec = gravity_vec.to(self.device)
            
        # 扩展重力向量以匹配环境数量
        gravity_vec = gravity_vec.unsqueeze(0).repeat(self.num_envs, 1)
        
        # 从四元数计算旋转矩阵
        R = self._quat_to_rotmat(base_quat)
        
        # 计算重力投影
        projected_gravity = torch.bmm(R, gravity_vec.unsqueeze(-1)).squeeze(-1)
        
        # 应用低通滤波到投影重力
        filtered_proj_gravity = self.alpha * projected_gravity + (1.0 - self.alpha) * self.last_projected_gravity
        self.last_projected_gravity = filtered_proj_gravity
        
        return filtered_proj_gravity

    @staticmethod
    def _quat_to_rotmat(quat):
        """
        将四元数转换为旋转矩阵
        """
        # 确保四元数是单位四元数
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        
        # 提取四元数分量
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # 构建旋转矩阵
        R = torch.zeros((quat.shape[0], 3, 3), device=quat.device)
        
        # 填充旋转矩阵元素
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R


class DepthImageViewer:
    def __init__(self, window_name="Depth Image", colormap=cv2.COLORMAP_TURBO):
        """
        初始化深度图像显示器
        
        Args:
            window_name (str): 窗口名称
            colormap: 颜色映射方案，可选值包括：
                     cv2.COLORMAP_JET       - 蓝红渐变
                     cv2.COLORMAP_VIRIDIS   - 绿紫渐变
                     cv2.COLORMAP_TURBO     - 改进的彩虹色
                     cv2.COLORMAP_PLASMA    - 紫橙渐变
                     cv2.COLORMAP_HOT       - 黑红黄白渐变
        """
        self.window_name = window_name
        self.colormap = colormap
        self.window_created = False
        
    def process_depth_image(self, depth_image):
        """
        处理深度图像并转换为彩色图
        
        Args:
            depth_image (torch.Tensor): 输入的深度图像张量
        Returns:
            numpy.ndarray: 处理后的彩色图像
        """
        try:
            # 转换为numpy数组
            if isinstance(depth_image, torch.Tensor):
                if depth_image.is_cuda:
                    depth_image = depth_image.cpu()
                depth_image = depth_image.numpy()
            
            # 添加偏移并归一化
            depth_image = depth_image #+ 0.75
            
            # 归一化到0-1范围
            depth_min = np.min(depth_image)
            depth_max = np.max(depth_image)
            if depth_max > depth_min:
                depth_norm = (depth_image - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = depth_image
                
            # 转换为uint8并应用颜色映射
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_uint8, self.colormap)
            
            # 添加深度值显示
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(depth_colormap, 
            #            f'Min: {depth_min:.2f}m', 
            #            (10, 20), 
            #            font, 
            #            0.5, 
            #            (255, 255, 255), 
            #            1)
            # cv2.putText(depth_colormap, 
            #            f'Max: {depth_max:.2f}m', 
            #            (10, 40), 
            #            font, 
            #            0.5, 
            #            (255, 255, 255), 
            #            1)
            
            return depth_colormap
            
        except Exception as e:
            print(f"处理深度图像时发生错误: {str(e)}")
            return None
    
    def display(self, depth_image):
        """
        显示彩色深度图像
        
        Args:
            depth_image (torch.Tensor): 要显示的深度图像
        """
        try:
            if not self.window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_created = True
            
            colorized_image = self.process_depth_image(depth_image)
            if colorized_image is not None:
                cv2.imshow(self.window_name, colorized_image)
                cv2.waitKey(1)
                
        except Exception as e:
            print(f"显示深度图像时发生错误: {str(e)}")
    
    def close(self):
        """关闭显示窗口"""
        try:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
        except Exception as e:
            print(f"关闭窗口时发生错误: {str(e)}")



class PointFoot:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_propriceptive_obs
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.num_terrain_map_obs = cfg.env.num_terrain_map_obs
        self.num_propriceptive_obs = cfg.env.num_propriceptive_obs

        if self.cfg.env.filtered_imu:
            self.imu_filter = IsaacGymImuFilter(
                num_envs=self.num_envs,
                device=self.device,
                alpha=self.cfg.env.filtered_alpha  # 可以调整这个值来改变滤波强度
            )

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        self.global_counter = 0

        

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._include_feet_height_rewards = self._check_if_include_feet_height_rewards()
        self._include_knee_height_rewards = self._check_if_include_knee_height_rewards()

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self._init_torque_buffer()

        
        self.pointcloud_buf = torch.zeros(self.num_envs, 5, 5, device=self.device, dtype=torch.float)

    def get_observations(self):
        return self.proprioceptive_obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def init_camera_warp(self, i):
        self.cam_pos_offset[i,0] = np.random.uniform(self.cfg.depth.position1[0], self.cfg.depth.position2[0])
        self.cam_pos_offset[i,1] = np.random.uniform(self.cfg.depth.position1[1], self.cfg.depth.position2[1])
        self.cam_pos_offset[i,2] = np.random.uniform(self.cfg.depth.position1[2], self.cfg.depth.position2[2])

        self.cam_rot_offset[i,1] = np.random.uniform(self.cfg.depth.angle[0], self.cfg.depth.angle[0])

        self.cam_fov[i] = np.random.uniform(self.cfg.depth.horizontal_fov[0], self.cfg.depth.horizontal_fov[1])
        

    def attach_camera(self, i , env_handle, actor_handle):
        if self.cfg.depth.use_camera and not self.cfg.depth.use_warp:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = torch.rand(1).item() * (self.cfg.depth.horizontal_fov[1] - self.cfg.depth.horizontal_fov[0]) + self.cfg.depth.horizontal_fov[0]
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            # self.camera_props.append(camera_props)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position1)
            camera_position[0] = np.random.uniform(config.position1[0], config.position2[0])
            camera_position[1] = np.random.uniform(config.position1[1], config.position2[1])  
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, camera_angle, 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.cfg.depth.use_warp:
            if not self.cfg.depth.use_camera:
                return

            if self.global_counter % self.cfg.depth.update_interval != 0:
                return
            # start_time = time.time()
            self.gym.step_graphics(self.sim) # required to render in headless mode
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            for i in range(self.num_envs):
                depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                    self.envs[i], 
                                                                    self.cam_handles[i],
                                                                    gymapi.IMAGE_DEPTH)
                
                depth_image = gymtorch.wrap_tensor(depth_image_)
                depth_image = self.process_depth_image(depth_image, i)
                if self.cfg.depth.invert:
                    depth_image = -depth_image  # 反向深度
                init_flag = self.episode_length_buf <= 1
                if init_flag[i]:
                    self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                else:
                # self.depth_buffer[i, 0] = depth_image.to(self.device).unsqueeze(0)
                    self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

            self.gym.end_access_image_tensors(self.sim)
        else:
            if self.global_counter % self.cfg.depth.update_interval != 0:
                return
            self.cam_pos = self.base_pos + quat_rotate(self.base_quat, self.cam_pos_offset)
            # print(self.cam_pos)
            # print(self.base_pos)
            self.cam_quat = quat_mul(self.base_quat, self.cam_quat_offset)
            depth_image = torch.zeros(self.num_envs, self.cfg.depth.original[1], self.cfg.depth.original[0], device=self.device)
            width = self.cfg.depth.resized[0]
            height = self.cfg.depth.resized[1]
            # start_time = time.time()
            for i in range(self.num_envs):
                
                with wp.ScopedDevice(self.device):
                    cam_pos = self.cam_pos[i]
                    cam_quat = self.cam_quat[i]
                    self.pixels = wp.zeros(width * height, dtype=float)
                    wp.launch(
                        kernel=depth_draw,
                        dim=width * height,
                        inputs=[self.mesh.id, cam_pos, cam_quat, width, height, self.cam_fov[i], self.pixels],
                    )
                    
                # pixels_np = self.pixels.to('cpu').numpy()  # 从 GPU 转到 CPU，并转换为 NumPy 数组

            # 将一维数组转换为二维数组 (height, width)，假设 87x58 是图像的分辨率
                pixels_reshaped = self.pixels.reshape((height, width))
                pixels_reshaped_tensor = torch.tensor(pixels_reshaped, dtype=torch.float32, device='cuda')
                # print(type(pixels_reshaped_tensor))
                # print(pixels_reshaped_tensor.shape)
                depth_image = pixels_reshaped_tensor
                depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
            
            # 如果你想保存图像，可以使用 cv2.imwrite
            # import matplotlib.pyplot as plt
            # # 使用 matplotlib 保存图像
            # plt.imshow(pixels_reshaped, cmap='gray')  # 使用灰度色图
            # plt.axis('off')  # 不显示坐标轴
            # plt.savefig('depth_image.png', bbox_inches='tight', pad_inches=0)  # 保存图像
            # plt.close()  # 关闭当前图像
                init_flag = self.episode_length_buf <= 1
                if init_flag[i]:
                    self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                else:
                # self.depth_buffer[i, 1] = depth_image.to(self.device).unsqueeze(0)
                    self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
            # print(time.time() - start_time)


        
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        #depth_image = self.normalize_depth_image(depth_image)
        return depth_image
    
    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image#[:-2, 4:-4]

    def visualize_pointcloud(self):
        """Visualize pointcloud if not headless"""
        if self.headless:
            return
            
        # Get pointcloud data
        # print(self.depth_buffer.shape)
        # points = self.depth_buffer[:,0,:,:]  # Only keep first element of second dimension (shape becomes [32,58,87])
        # for env_idx in range(self.num_envs):
        #     depth_image = -self.depth_buffer[env_idx, -1].double()
        #     height, width = depth_image.shape
        #     fx = width / (2 * np.tan(self.cfg.depth.horizontal_fov / 2))
        #     fy = fx
        #     cx, cy = width / 2, height / 2
            
        #     y, x = torch.meshgrid(torch.arange(height, device=self.device, dtype=torch.float64), 
        #                         torch.arange(width, device=self.device, dtype=torch.float64), 
        #                         indexing='ij')
        #     x = (x - cx) / fx
        #     y = (y - cy) / fy
            
        #     z = depth_image
        #     points = torch.stack([x * z, y * z, z], dim=-1).reshape(-1, 3)
            
        #     camera_pose = self.gym.get_camera_transform(self.sim, self.envs[env_idx], self.cam_handles[env_idx])
        #     quat = camera_pose.r
        #     rot_matrix = np.array([
        #         [1 - 2*quat.y*quat.y - 2*quat.z*quat.z, 2*quat.x*quat.y - 2*quat.z*quat.w, 2*quat.x*quat.z + 2*quat.y*quat.w],
        #         [2*quat.x*quat.y + 2*quat.z*quat.w, 1 - 2*quat.x*quat.x - 2*quat.z*quat.z, 2*quat.y*quat.z - 2*quat.x*quat.w],
        #         [2*quat.x*quat.z - 2*quat.y*quat.w, 2*quat.y*quat.z + 2*quat.x*quat.w, 1 - 2*quat.x*quat.x - 2*quat.y*quat.y]
        #     ], dtype=np.float64)
            
        #     camera_rotation = torch.tensor(rot_matrix, device=self.device, dtype=torch.float64)
        #     camera_translation = torch.tensor([camera_pose.p.x, camera_pose.p.y, camera_pose.p.z], 
        #                                     device=self.device, dtype=torch.float64)
            
        #     points = (camera_rotation @ points.T).T + camera_translation
        #     points = points.cpu().numpy()
            
        #     # Convert points to vertices array
        #     verts = []
        #     for i in range(height - 1):
        #         for j in range(width - 1):
        #             idx = i * width + j
        #             p1 = points[idx]
        #             p2 = points[idx + 1]
        #             p3 = points[idx + width]
                    
        #             # Only add vertices if depth is valid (non-zero)
        #             if depth_image[i,j] > 0 and depth_image[i,j+1] > 0:
        #                 verts.append([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]])
        #             if depth_image[i,j] > 0 and depth_image[i+1,j] > 0:
        #                 verts.append([p1[0], p1[1], p1[2], p3[0], p3[1], p3[2]])

        #     verts = np.array(verts, dtype=np.float32)
        #     if len(verts) > 0:
        #         # Create line list geometry
        #         for v in verts:
        #             gymutil.draw_line(gymapi.Vec3(v[0], v[1], v[2]), 
        #                             gymapi.Vec3(v[3], v[4], v[5]),
        #                             gymapi.Vec3(0.0, 1.0, 0.0),
        #                             self.gym, self.viewer, self.envs[env_idx])


    def _init_torque_buffer(self):
        # 使用零初始化延迟缓冲区
        self.max_delay_steps = int(self.cfg.domain_rand.max_step_delay_time/self.sim_params.dt)
        # 初始化torques缓冲区，每个环境都有一个独立的缓冲区
        self.torques_buffer = torch.zeros((self.max_delay_steps, self.num_envs, self.cfg.env.num_actions), device=self.device)
        # 初始化每个环境的随机延迟步数
        if(self.cfg.domain_rand.randomize_step_delay):
            self.delay_steps = torch.randint(0, self.max_delay_steps, (self.num_envs,), device=self.device)
            # 初始化固定大小的缓冲区
        self.current_step = 0  # 当前时间步索引

        if self.cfg.domain_rand.random_motor_strength:
            str_rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
            # self.motor_random_scale = rand_range[0] + (rand_range[1] - rand_range[0]) * torch.rand_like(self.torques)
        else:
            self.motor_strength = torch.ones_like(self.torques)  

        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets = torch.rand(self.num_envs, self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.global_counter += 1
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.last_lin_reward = self._reward_tracking_lin_vel()
        
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            current_torques = self._compute_torques(self.actions).view(self.torques.shape)
            if(self.cfg.domain_rand.randomize_step_delay):
                # 将当前计算的torques添加到缓冲区
                # self.torques_buffer = torch.cat((self.torques_buffer, current_torques.unsqueeze(0)), dim=0)

                # # 从缓冲区中取出延迟的torques
                # delayed_indices = torch.clamp(self.torques_buffer.shape[0] - self.delay_steps - 1, min=0)
                # indices = torch.arange(self.num_envs)
                # self.torques = self.torques_buffer[delayed_indices, indices]

                # # 保持缓冲区大小，删除最早的torques
                # if self.torques_buffer.shape[0] > self.max_delay_steps:
                #     self.torques_buffer = self.torques_buffer[1:]
                    # 存储当前扭矩到缓冲区的当前位置
                self.torques_buffer[self.current_step] = current_torques

                # 计算每个环境的延迟索引
                delayed_indices = (self.current_step - self.delay_steps) % self.max_delay_steps
                indices = torch.arange(self.num_envs)

                # 提取对应的延迟扭矩
                self.torques = self.torques_buffer[delayed_indices, indices]

                # 更新当前时间步索引
                self.current_step = (self.current_step + 1) % self.max_delay_steps
            else:
                self.torques = current_torques

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        if(self.headless == False):
            self._update_visualization()
            
        if self.cfg.depth.use_camera:
            self.gym.end_access_image_tensors(self.sim)
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.proprioceptive_obs_buf = torch.clip(self.proprioceptive_obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        # print(self.depth_buffer[:, -2])
        # print(self.global_counter)
        # print(self.extras["depth"])
        return self.proprioceptive_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _create_quat(self, angle, axis='x'):
        """根据给定的轴和角度生成四元数"""
        half_angle = angle / 2
        sin_half_angle = torch.sin(half_angle)

        if axis == 'x':
            return torch.stack([torch.cos(half_angle), sin_half_angle, torch.zeros_like(angle), torch.zeros_like(angle)], dim=-1)
        elif axis == 'y':
            return torch.stack([torch.cos(half_angle), torch.zeros_like(angle), sin_half_angle, torch.zeros_like(angle)], dim=-1)

    def _quat_multiply(self, q, r):
        """四元数相乘"""
        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # self.rigid_body_states[:, self.base_index, 0:3]-=0.1
        # self.root_states[:, 2:3]-=0.1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        # print(self.base_pos)
        # print(self.terrain_types)
        
        self.base_quat[:] = self.root_states[:, 3:7]
        
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.domain_rand.random_imu_offset:
            base_quat_bias = self._quat_multiply(self._quat_multiply(self.base_quat, self.roll_quat), self.pitch_quat)

            self.projected_gravity_bias[:] = quat_rotate_inverse(base_quat_bias, self.gravity_vec)
        else:
            self.projected_gravity_bias[:] = self.projected_gravity[:]
            
        if self.cfg.terrain.mesh_type == "trimesh":
            self.terrain_type[:] = self.terrain.get_terrain_type(self.base_pos).to(self.device) 
            # Normalize terrain_type from [1,5] to [0,1]
        self.terrain_type_normalized = (self.terrain_type - 1.0) / 4.0
        
        expert_labels = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        # 地形0,1用专家0
        mask_expert0 = (self.terrain_type <= 2).squeeze(1)  
        expert_labels[mask_expert0, 0] = 1
        
        # 地形2用专家1  
        mask_expert1 = (self.terrain_type == 3).squeeze(1)
        expert_labels[mask_expert1, 1] = 1
        mask_expert1 = (self.terrain_type == 4).squeeze(1)
        expert_labels[mask_expert1, 1] = 1
        
        # 地形3,4,5用专家2
        mask_expert2 = (self.terrain_type >= 5).squeeze(1)
        expert_labels[mask_expert2, 2] = 1

        self.expert_labels = expert_labels.clone()
        self.extras["expert_labels"] = expert_labels.clone()
        self.extras["terrain_type"] = self.terrain_type_normalized.clone()


        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()

            self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        self._compute_feet_states()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.lastbase_lin_vel[:] = self.base_lin_vel[:]

        if self.viewer and self.enable_viewer_sync:# and self.debug_viz:
            # self._draw_debug_vis()
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            # self._draw_goals()
            # self._draw_feet()
            if self.cfg.depth.use_camera:
                self.lookat_id = 0
                viewer = DepthImageViewer(window_name="warp" ,colormap=cv2.COLORMAP_HOT)  
                viewer.display(self.depth_buffer[self.lookat_id, -1])

                # window_name = "Depth Image"
                # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy()+ 0.75)
                # cv2.waitKey(1)

    def _check_if_include_feet_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "feet_height" in scale:
                return True
        return False
    
    def _check_if_include_knee_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "knee_height" in scale:
                return True
        return False

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample(env_ids)

        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_air_time[env_ids] = 0.
        self.current_max_feet_height[env_ids] = 0.
        self.last_max_feet_height[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.compute_proprioceptive_observations()
        self.compute_privileged_observations()

        self._add_noise_to_obs()

    def _add_noise_to_obs(self):
        # add noise if needed
        if self.add_noise:
            obs_noise_vec, privileged_extra_obs_noise_vec = self.noise_scale_vec
            obs_noise_buf = (2 * torch.rand_like(self.proprioceptive_obs_buf) - 1) * obs_noise_vec
            self.proprioceptive_obs_buf += obs_noise_buf
            if self.num_privileged_obs is not None:
                privileged_extra_obs_buf = (2 * torch.rand_like(
                    self.privileged_obs_buf[:, len(self.noise_scale_vec[0]):]) - 1) * privileged_extra_obs_noise_vec
                if self.cfg.noise.add_pri_noise:
                    self.privileged_obs_buf += torch.cat((obs_noise_buf, privileged_extra_obs_buf), dim=1)

    def compute_privileged_observations(self):
        if self.num_privileged_obs is not None:
            self._compose_privileged_obs_buf_no_height_measure()
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights_critic:
                self.privileged_obs_buf = self._add_height_measure_to_buf(self.privileged_obs_buf)

            if self.cfg.env.measure_lin_vel_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.base_lin_vel * self.obs_scales.lin_vel),
                                                    dim=-1)
            
                
            if self.cfg.env.measure_feet_height_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.feet_height), dim=-1)
                
            if self.cfg.env.measure_base_height_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.base_height.unsqueeze(1)), dim=-1)

            if self.cfg.env.measure_contact_force_critic:
                # 将 self.contact_forces 转换为 2D 张量，形状变为 [num_envs, num_feet * 3]
                contact_forces_flat = torch.flatten(self.contact_forces[:, self.feet_indices, :], start_dim=1)
                # 现在 self.privileged_obs_buf 和 contact_forces_flat 都是 2D 张量，可以拼接
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact_forces_flat), dim=-1)

            

            if self.cfg.env.measure_contact_filt_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.contact_filt), dim=-1)
         
            if self.cfg.env.measure_friction_coeff_critic:
                # 将 friction_coeffs 转换为 2D 张量
                friction_coeffs_flat = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
                # 然后进行拼接
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, friction_coeffs_flat), dim=-1)
            
            if self.cfg.env.measure_step_delay:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.delay_steps.unsqueeze(-1)), dim=-1)

            if self.cfg.env.measure_restitutions_critic:
                restitutions_coeffs_flat = self.restitution.to(self.device).to(torch.float).squeeze(-1)
                # 然后进行拼接
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, restitutions_coeffs_flat), dim=-1)

            if self.cfg.env.measure_gravityoffset_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.gravities), dim=-1)    

            if self.cfg.env.measure_motor_strength:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.motor_strength[0]), dim=-1)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.motor_strength[1]), dim=-1)

            if self.cfg.env.measure_motor_offset_critic:
                motor_offset = (self.motor_offsets - self.cfg.domain_rand.motor_offset_range[0])/\
                    (self.cfg.domain_rand.motor_offset_range[1] - self.cfg.domain_rand.motor_offset_range[0])
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.motor_offsets), dim=-1)

            if self.cfg.env.measure_terrain_type_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.terrain_type), dim=-1)

            if self.cfg.env.measure_link_mass_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.all_link_mass), dim=-1)

            if self.cfg.env.measure_target_height_critic:
                target = self.cfg.rewards.base_height_target*torch.ones(self.num_envs, 1, device=self.device, dtype=torch.float)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, target), dim=-1   )
            
            if self.cfg.env.measure_target_feet_height_critic:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.target_feet_height), dim=-1)

            if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
                raise RuntimeError(
                    f"privileged_obs_buf size ({self.privileged_obs_buf.shape[1]}) does not match num_privileged_obs ({self.num_privileged_obs})")

    def _compose_privileged_obs_buf_no_height_measure(self):
        self.privileged_obs_buf = torch.cat((self.filtered_ang_vel * self.obs_scales.ang_vel,
                                             self.filtered_gravity * self.obs_scales.gravity,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions,
                                             self.commands[:, :3] * self.commands_scale,
                                             self.ref_contact,
                                            #  self.target_feet_height,
                                             ), dim=-1)

    def compute_proprioceptive_observations(self):
        
        self._compose_proprioceptive_obs_buf_no_height_measure()
        if self.cfg.terrain.measure_heights_actor:
            self.proprioceptive_obs_buf = self._add_height_measure_to_buf(self.proprioceptive_obs_buf)
        if self.proprioceptive_obs_buf.shape[1] != self.num_obs:
            raise RuntimeError(
                f"obs_buf size ({self.proprioceptive_obs_buf.shape[1]}) does not match num_obs ({self.num_obs})")

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf

    def _compose_proprioceptive_obs_buf_no_height_measure(self):
        # 应用滤波
        if self.cfg.env.filtered_imu:
            self.filtered_ang_vel = self.imu_filter.filter_ang_vel(self.base_ang_vel)
            self.filtered_gravity = self.imu_filter.filter_orientation(self.base_quat)
        else:
            self.filtered_ang_vel = self.base_ang_vel
            self.filtered_gravity = self.projected_gravity_bias
        self.proprioceptive_obs_buf = torch.cat((self.filtered_ang_vel * self.obs_scales.ang_vel,
                                                 self.filtered_gravity * self.obs_scales.gravity,
                                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                 self.dof_vel * self.obs_scales.dof_vel,
                                                 self.actions,
                                                 self.commands[:, :3] * self.commands_scale,
                                                 self.ref_contact,
                                                #  self.target_feet_height,
                                                 ), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.device)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                re_bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1),
                                                        device='cpu')
                self.restitution = restitution_buckets[re_bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
             # 应用 motor_strength 的阈随机化
            # if self.cfg.domain_rand.random_motor_strength:
            #     min_factor = self.cfg.domain_rand.motor_strength_range[0]
            #     max_factor = self.cfg.domain_rand.motor_strength_range[1]
            #     # 生成每个DOF的随机因子
            #     random_factors = torch.rand(self.num_dof, device=self.device) * (max_factor - min_factor) + min_factor
            #     # 应用随机因子到 torque_limits
            #     self.torque_limits = self.torque_limits * random_factors
                
            #     # 更新 props 中的 effort 属性
            #     props["effort"] = self.torque_limits.cpu().numpy()
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            self.base_mass[env_id] = props[0].mass

        if self.cfg.domain_rand.randomize_link_mass:
            for link_index in range(1, len(props)):
                # 获取当前链接的原始质量
                original_mass = props[link_index].mass
                randomize_link_rate = self.cfg.domain_rand.randomize_link_rate
                random_rate = torch.rand(1).item() * (randomize_link_rate[1] - randomize_link_rate[0]) + randomize_link_rate[0]
                # 应用随机比例到质量上
                new_mass = original_mass * (1.0 + random_rate)

                # 设置新的质量
                props[link_index].mass = new_mass
                
            for link_index in range(len(props)):
                self.all_link_mass[env_id][link_index] = props[link_index].mass
                

        if self.cfg.domain_rand.randomize_base_com:
            com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
            props[0].com.x += np.random.uniform(-com_x, com_x)
            props[0].com.y += np.random.uniform(-com_y, com_y)
            props[0].com.z += np.random.uniform(-com_z, com_z)

        if self.cfg.domain_rand.randomize_link_com:
            for link_index in range(1, len(props)):
                com_x, com_y, com_z = self.cfg.domain_rand.rand_link_vec
                props[link_index].com.x += np.random.uniform(-com_x, com_x)
                props[link_index].com.y += np.random.uniform(-com_y, com_y)
                props[link_index].com.z += np.random.uniform(-com_z, com_z)

         # randomize inertia of all body
        if self.cfg.domain_rand.random_inertia:
            rng = self.cfg.domain_rand.inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xx[env_id, s] = rd_num
                props[s].inertia.x.x *= rd_num
                
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xy[env_id, s] = rd_num
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xz[env_id, s] = rd_num
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yy[env_id, s] = rd_num
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yz[env_id, s] = rd_num
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_zz[env_id, s] = rd_num
                props[s].inertia.z.z *= rd_num

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample(self, env_ids):
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)    

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


                # 设置 Beta 分布的参数，调整采样偏向
        # beta_temp = self.cfg.commands.beta
        # alpha, beta = beta_temp, beta_temp  # alpha < beta 表示采样值更偏向于0

        # # 线速度 x 方向的采样
        # speed_range_x = self.command_ranges["lin_vel_x"]
        # speed_scale_x = torch.distributions.Beta(alpha, beta).sample((len(env_ids),)).to(self.device)
        # speed_sample_x = speed_scale_x * (speed_range_x[1] - speed_range_x[0]) + speed_range_x[0]
        # self.commands[env_ids, 0] = speed_sample_x

        # # 线速度 y 方向的采样
        # speed_range_y = self.command_ranges["lin_vel_y"]
        # speed_scale_y = torch.distributions.Beta(alpha, beta).sample((len(env_ids),)).to(self.device)
        # speed_sample_y = speed_scale_y * (speed_range_y[1] - speed_range_y[0]) + speed_range_y[0]
        # self.commands[env_ids, 1] = speed_sample_y

        # # 角速度的采样
        # if self.cfg.commands.heading_command:
        #     heading_range = self.command_ranges["heading"]
        #     heading_scale = torch.distributions.Beta(alpha, beta).sample((len(env_ids),)).to(self.device)
        #     heading_sample = heading_scale * (heading_range[1] - heading_range[0]) + heading_range[0]
        #     self.commands[env_ids, 3] = heading_sample
        # else:
        #     ang_vel_yaw_range = self.command_ranges["ang_vel_yaw"]
        #     ang_vel_yaw_scale = torch.distributions.Beta(alpha, beta).sample((len(env_ids),)).to(self.device)
        #     ang_vel_yaw_sample = ang_vel_yaw_scale * (ang_vel_yaw_range[1] - ang_vel_yaw_range[0]) + ang_vel_yaw_range[0]
        #     self.commands[env_ids, 2] = ang_vel_yaw_sample

        # # 设置小命令为零
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.15).unsqueeze(1)


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.random_motor_strength:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
              
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots."""
        max_push_force = (
                self.base_mass.mean().item()
                * self.cfg.domain_rand.max_push_vel_xy
                / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[:, 0, 0:3] = quat_rotate(
            self.base_quat, rigid_body_external_forces
        )
        self.rigid_body_external_forces[:, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2.0
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        obs_noise_vec = torch.zeros(self.cfg.env.num_propriceptive_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # obs_noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        obs_noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        obs_noise_vec[3:6] = noise_scales.gravity * noise_level
        dof_pos_end_idx = 6 + self.num_dof
        obs_noise_vec[6:dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        dof_vel_end_idx = dof_pos_end_idx + self.num_dof
        obs_noise_vec[dof_pos_end_idx:dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        last_action_end_idx = dof_vel_end_idx + self.num_actions
        obs_noise_vec[dof_vel_end_idx:last_action_end_idx] = 0.  # previous actions
        command_end_idx = last_action_end_idx + self.cfg.commands.num_commands
        obs_noise_vec[last_action_end_idx:command_end_idx] = 0.  # commands
        if self.cfg.env.num_privileged_obs is not None:
            privileged_extra_obs_noise_vec = torch.zeros(
                self.cfg.env.num_privileged_obs - self.cfg.env.num_propriceptive_obs, device=self.device)
        else:
            privileged_extra_obs_noise_vec = None

        if self.cfg.terrain.measure_heights_actor:
            measure_heights_end_idx = command_end_idx + len(self.cfg.terrain.measured_points_x) * len(
                self.cfg.terrain.measured_points_y)
            obs_noise_vec[
            command_end_idx:measure_heights_end_idx] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights_critic:
            if self.cfg.env.num_privileged_obs is not None:
                privileged_extra_obs_noise_vec[
                0:len(self.cfg.terrain.measured_points_x) * len(
                    self.cfg.terrain.measured_points_y)] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.env.measure_lin_vel_critic:
            terrain_idx = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
            lin_vel_idx = terrain_idx + 3
            privileged_extra_obs_noise_vec[terrain_idx:lin_vel_idx] \
                                                = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        
        if self.cfg.env.measure_feet_height_critic:
            feet_height_idx = lin_vel_idx + 2
            privileged_extra_obs_noise_vec[lin_vel_idx:feet_height_idx] \
                                                = noise_scales.feet_height * noise_level * self.obs_scales.feet_height
        else:
            feet_height_idx = lin_vel_idx

        if self.cfg.env.measure_base_height_critic:
            base_height_idx = feet_height_idx + 1
            privileged_extra_obs_noise_vec[feet_height_idx:base_height_idx] \
                                                = noise_scales.base_height * noise_level * self.obs_scales.base_height
        else:
            base_height_idx = feet_height_idx
        
        if self.cfg.env.measure_contact_force_critic:
            contact_force_idx = base_height_idx + 3
            privileged_extra_obs_noise_vec[base_height_idx:contact_force_idx] \
                                                = noise_scales.contact_force * noise_level * self.obs_scales.contact_force
        else:
            contact_force_idx = base_height_idx 
        
        if self.cfg.env.measure_contact_filt_critic:
            contact_filt_idx = contact_force_idx + 1
            privileged_extra_obs_noise_vec[contact_force_idx:contact_filt_idx] \
                                                = noise_scales.contact_filt * noise_level * self.obs_scales.contact_filt
        else:
            contact_filt_idx = contact_force_idx


        if self.cfg.env.measure_friction_coeff_critic:
            friction_coeff_idx = contact_filt_idx + 3*2
            privileged_extra_obs_noise_vec[contact_filt_idx:friction_coeff_idx] \
                                                = noise_scales.friction_coeff * noise_level * self.obs_scales.friction_coeff
        else:
            friction_coeff_idx = contact_filt_idx  
            
        return obs_noise_vec, privileged_extra_obs_noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.root_states[:,2:3]-=0.1
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, 0:3]

        self.cam_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.cam_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.cam_quat[:, 3] = 1

        self.terrain_type = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.terrain_type_normalized = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.random_imu_offset:
            # 初始化随机偏置（以弧度为单位）
            imu_range = self.cfg.domain_rand.random_imu_range
            self.roll_bias = torch.rand(self.num_envs, device=self.device) * torch.pi*(imu_range[1]-imu_range[0]) + torch.pi*imu_range[0]  
            self.pitch_bias = torch.rand(self.num_envs, device=self.device)* torch.pi*(imu_range[1]-imu_range[0]) + torch.pi*imu_range[0]  
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        # self.rigid_body_states[:, self.base_index, 2:3]-=0.1
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.knee_state = self.rigid_body_states[:, self.knee_indices, :]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        if self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)
            sim_params = self.gym.get_sim_params(self.sim)
            gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.81]).to(self.device)
            self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
            sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
            self.gym.set_sim_params(self.sim, sim_params)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.motor_random_scale = torch.ones_like(self.torques)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.last_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)

        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.first_air = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                      device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.current_max_feet_pos_x = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.last_max_knee_height = torch.zeros(self.num_envs, self.knee_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_knee_height = torch.zeros(self.num_envs, self.knee_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.ref_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.target_feet_height = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device,
                                         requires_grad=False)*self.cfg.rewards.max_feet_height
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.lastbase_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        
        if self.cfg.domain_rand.random_imu_offset:
            # 生成 roll 偏置四元数
            self.roll_quat = self._create_quat(self.roll_bias, axis='x')
            
            # 生成 pitch 偏置四元数
            self.pitch_quat = self._create_quat(self.pitch_bias, axis='y')

            # 将偏置应用到 base_quat 上（通过四元数乘法）
            base_quat_bias = self._quat_multiply(self._quat_multiply(self.base_quat, self.roll_quat), self.pitch_quat)

            self.projected_gravity_bias = quat_rotate_inverse(base_quat_bias, self.gravity_vec)
        else:
            self.projected_gravity_bias = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.expert_labels = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.expert_labels[:,0] = 1

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        if self.cfg.depth.use_warp:
            wp.init()
            self.wprender = wprender.UsdRenderer("/home/chenfu/Downloads/USD/test.usd")
            warp_vertices = wp.array(self.terrain.vertices, dtype=wp.vec3)
            
            warp_triangles = wp.array(self.terrain.triangles.flatten(order='C'), dtype=int)
            # print(self.terrain.vertices)
            # print(self.terrain.triangles.flatten(order='C').shape)
            # print(self.terrain.heightsamples)
            self.mesh = wp.Mesh(points=warp_vertices, indices=warp_triangles)

            # self.wprender.begin_frame(0.1)
            # self.wprender.render_mesh(
            #     name="mesh",
            #     points=self.mesh.points.numpy(),
            #     indices=self.mesh.indices.numpy(),
            #     colors=((0.35, 0.55, 0.9),) * len(self.mesh.points),
            # )
            # self.wprender.end_frame()
            
            # import matplotlib.pyplot as plt
            # plt.imshow(-self.pixels.numpy().reshape(58, 87), cmap="gray")

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)
        
    def _update_visualization(self):
        import numpy as np
        from scipy.spatial.transform import Rotation as R  # 用于处理四元数旋转

        # print("terrain:",self.terrain_type)

        self.gym.clear_lines(self.viewer)
        if self.cfg.depth.use_camera:
            self.visualize_pointcloud()
        # 遍历所有环境
        for i in range(self.num_envs):
            # 获取 base_link 的速度和姿态
            linear_velocity = self.commands[i, 0:2].cpu().numpy()
            yaw_rate = self.commands[i, 2].cpu().numpy()
            base_position = self.rigid_body_states[i, self.base_index, 0:3].cpu().numpy()  # [x, y, z]
            base_orientation = self.rigid_body_states[i, self.base_index, 3:7].cpu().numpy()  # [qx, qy, qz, q
            # 将四元数转换为旋转矩阵
            rotation = R.from_quat(base_orientation)

            # 将线速度转换到世界坐标系
            rotated_linear_velocity = rotation.apply(np.array([linear_velocity[0], linear_velocity[1], 0.0]))

            # 计算新位置，考虑线速度在世界坐标系中的影响
            new_x = base_position[0] + rotated_linear_velocity[0]  / 3
            new_y = base_position[1] + rotated_linear_velocity[1]  / 3
            new_z = base_position[2] # 如果没有垂直运动，z 位置保持不变

            # 可视化 base_link 的新位置（期望位置）
            sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))
            target_vec = gymapi.Transform(gymapi.Vec3(new_x, new_y, new_z), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], target_vec)

            # # 可视化 camera 的位置
            # sphere_geom_cam = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
            # target_vec = gymapi.Transform(gymapi.Vec3(self.cam_pos[i,0], self.cam_pos[i,1], self.cam_pos[i,2]), r=None)
            # gymutil.draw_lines(sphere_geom_cam, self.gym, self.viewer, self.envs[i], target_vec)

            # cam_position = self.cam_pos[i].cpu().numpy()  # [x, y, z]
            # # 获取相机的旋转矩阵
            # cam_quat = self.cam_quat[i].cpu().numpy()  # [qx, qy, qz, qw]
            # rotation = R.from_quat(cam_quat)
            
            # # 定义相机坐标系中的 x 轴方向向量
            # cam_x_axis = np.array([0.2, 0.0, 0.0])  # x轴方向
            
            # # 使用旋转矩阵将相机的 x 轴旋转到世界坐标系
            # rotated_cam_x_axis = rotation.apply(cam_x_axis)

            # # 可视化相机的 x 轴方向（从相机位置出发）
            # sphere_geom_cam_x = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))  # 小球可视化
            # target_vec_cam_x = gymapi.Transform(gymapi.Vec3(cam_position[0] + rotated_cam_x_axis[0],
            #                                                 cam_position[1] + rotated_cam_x_axis[1],
            #                                                 cam_position[2] + rotated_cam_x_axis[2]), r=None)
            
            # # 绘制相机 x 轴方向的线
            # gymutil.draw_lines(sphere_geom_cam_x, self.gym, self.viewer, self.envs[i], target_vec_cam_x)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        print(asset_path)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.inertia_mask_xx = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_zz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        hip_names = [s for s in body_names if self.cfg.asset.hip_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.camera_props = []
        # self.cam_tensors = []
        self.cam_pos_offset = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.cam_rot_offset = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.cam_fov = 87*torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            # pos[2:3] +=0.12
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            self.all_link_mass = torch.zeros(self.num_envs, len(body_props),dtype=torch.float,device=self.device, requires_grad=False)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            self.attach_camera(i, env_handle, actor_handle)

            self.init_camera_warp(i)
            self.cam_quat_offset = quat_from_euler_xyz(self.cam_rot_offset[:,0], self.cam_rot_offset[:,1],self.cam_rot_offset[:,2])

        base_name = [s for s in body_names if self.cfg.asset.base_name in s][0]
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
            
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         knee_names[i])
        hip_names = ["hip_L_Joint", "hip_R_Joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        # print(self.hip_indices)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights_actor and not self.terrain.cfg.measure_heights_critic:
            return
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_below_foot(self):
        """ Samples heights of the terrain at required points around each foot.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.feet_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_heights_below_knee(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.knee_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.knee_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

    def _compute_feet_states(self):
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.last_feet_air_time = self.feet_air_time * self.first_contact + self.last_feet_air_time * ~self.first_contact
        # 当脚抬起时，更新 last_contact_time
        self.last_contact_time = self.contact_time * self.first_air + self.last_contact_time * ~self.first_air

        self.feet_air_time *= ~self.contact_filt
        self.contact_time *= self.contact_filt
        if self._include_feet_height_rewards:
            self.last_max_feet_height = self.current_max_feet_height * self.first_contact + self.last_max_feet_height * ~self.first_contact
            self.current_max_feet_height *= ~self.contact_filt
            self.feet_height = self.feet_state[:, :, 2] - self._get_heights_below_foot()
            self.current_max_feet_height = torch.max(self.current_max_feet_height,
                                                     self.feet_height)

        if self._include_knee_height_rewards:    
            self.knee_state = self.rigid_body_states[:, self.knee_indices, :]
            self.last_max_knee_height = self.current_max_knee_height * self.first_contact + self.last_max_knee_height * ~self.first_contact
            self.current_max_knee_height *= ~self.contact_filt
            # 计算膝关节的高度
            self.knee_height = self.knee_state[:, :, 2] - self._get_heights_below_foot()
            # 更新膝关节的当前最大高度
            self.current_max_knee_height = torch.max(self.current_max_knee_height, self.knee_height)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.first_air = (self.contact_time > 0.) * ~self.contact_filt
        self.feet_air_time += self.dt

        self.contact_time += self.dt

        
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        sin_pos = torch.sin(2 * torch.pi * phase)
        cos2_pos = torch.cos(4*torch.pi*phase)
        self.ref_contact = torch.zeros_like(self.contact_filt)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        cos2_pos_l = cos2_pos.clone()
        cos2_pos_r = cos2_pos.clone()
        max_feet_height=self.cfg.rewards.max_feet_height
        # Reshape terrain_type from [8192,1] to [8192,2]
        terrain_type = self.terrain_type.clone()
        terrain_type = terrain_type.repeat(1, 2)
        # self.target_feet_height[:, 0] = 0.034+torch.max(-self.cfg.rewards.max_feet_height*sin_pos_r, torch.zeros_like(sin_pos_r))
        # self.target_feet_height[:, 1] = 0.034+torch.max(self.cfg.rewards.max_feet_height*sin_pos_l, torch.zeros_like(sin_pos_l))
        self.target_feet_height[:, 0] = torch.max(max_feet_height*(1 - cos2_pos_r)/2, torch.zeros_like(cos2_pos_r))
        self.target_feet_height[:, 1] = torch.max(max_feet_height*(1 - cos2_pos_l)/2, torch.zeros_like(cos2_pos_l))

        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_l[sin_pos_l < 0] = 1
        sin_pos_r[sin_pos_r > 0] = 1
        self.ref_contact[:, 0] = sin_pos_r
        self.ref_contact[:, 1] = sin_pos_l
        self.target_feet_height = self.target_feet_height * ~self.ref_contact + 0.03
        self.ref_contact[torch.abs(sin_pos) < 0.0000001] = 1

        
        
    def quat_to_euler(self, quat):
        """
        将四元数转换为欧拉角 (roll, pitch, yaw)，返回 shape 为 [num_envs, 3] 的 tensor。
        假设输入四元数的顺序为 (w, x, y, z)。
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # 计算 roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # 计算 pitch
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(sinp.clamp(-1, 1))

        # 计算 yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
            # 提取 roll 和 pitch 角度
        roll, pitch, _ = self.quat_to_euler(self.base_quat)

        # 根据 roll 和 pitch 计算惩罚权重
        # roll 和 pitch 越接近 0（水平），惩罚越大
        attitude_penalty_weight = torch.exp(-torch.abs(roll)/0.5 - torch.abs(pitch)/0.5)  # 指数衰减函数
        
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) #* attitude_penalty_weight

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_tracking_orientation(self):
        # Penalize non flat base orientation
        orientation_err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-orientation_err/0.1)
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        # contact = self.contact_forces[:, self.feet_indices, 2] > 2.
        contact = self.contact_filt
        stance_mask = self.ref_contact
        reward = torch.where(contact == stance_mask, 1.0, -1.0)
        return torch.mean(reward, dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # 
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        reward = torch.square(self.base_height - self.cfg.rewards.base_height_target)
        final_reward = torch.where(self.base_height > self.cfg.rewards.base_height_target, reward*1.5, reward)
        return final_reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.min_feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime
    
    def _reward_last_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.cycle_time/2,
                     torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime
    
    def _reward_last_contact_time(self):
        # Reward steps between proper duration
        rew_contact_below_min = torch.sum(
            torch.min(self.contact_time - self.cfg.rewards.cycle_time/2, 
                     torch.zeros_like(self.contact_time)) * self.first_air,
            dim=1)
        rew_contact_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.contact_time,
                      torch.zeros_like(self.contact_time)) * self.first_air,
            dim=1)
        rew_contact_time = rew_contact_below_min + rew_contact_above_max
        return rew_contact_time
    
    def _reward_lin_vel_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_tracking_lin_vel() - self.last_lin_reward)
        detalvel = -torch.sum(torch.abs(self.base_lin_vel-self.lastbase_lin_vel),dim=-1)
        return detalvel/self.dt
        return delta_phi/self.dt

    def _reward_no_fly(self):
        # contacts = self.contact_forces[:, self.feet_indices, 2] > 0.2
        # single_contact = torch.sum(1. * contacts, dim=1) == 1
        # return 1. * single_contact

        left_foot_contact = self.contact_filt[:, 0]  # 左脚接触状态
        right_foot_contact = self.contact_filt[:, 1]  # 右脚接触状态

        # 比较两只脚的接触状态是否相同
        same_state = left_foot_contact == right_foot_contact

        # 如果状态相同，给予惩罚-1.0；如果状态不同，给予1.0
        reward = torch.where(same_state, -1.0, 1.0)

        # 返回每个环境的奖励值
        return reward
    
    def _reward_tracking_base_height(self):
    # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.exp(-torch.square(base_height - self.cfg.rewards.base_height_target)/0.15)

    def _reward_unbalance_feet_air_time(self):
        return torch.var(self.last_feet_air_time, dim=-1)

    def _reward_unbalance_feet_height(self):
        return torch.var(self.last_max_feet_height, dim=-1)
    
    def _reward_unbalance_knee_height(self):
        return torch.var(self.last_max_knee_height, dim=-1)
    
    def _reward_target_feet_height(self):
        # reward = torch.max(self.feet_height-self.cfg.rewards.max_feet_height-0.03, \
        #           torch.zeros_like(self.feet_height))
    

        # adaptive_reward_feet = torch.mean(reward, dim=1)
        # return adaptive_reward_feet
        # 计算足端最大高度的惩罚
        reward_feet = torch.abs(self.feet_height - self.target_feet_height)
        adaptive_reward_feet = torch.mean(reward_feet, dim=1)
        return adaptive_reward_feet#*rate_terrain
    
    def _reward_lift_feet_height(self):
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1) + 0.01
        speed_weight = torch.exp(-1/speed)
        feet_height = torch.clamp(self.last_max_feet_height, max=0.13)
        return torch.mean(speed_weight.unsqueeze(1) * feet_height, dim=1)
    
    def _reward_hip_pos(self):
        # print("hip_indices:", self.hip_indices)
        # print("default_dof_pos shape:", self.default_dof_pos.shape)
        # print(self.dof_pos.shape)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize displacement and rotation at zero commands
        reward_lin = torch.abs(self.base_lin_vel[:, :2]) * (torch.abs(self.commands[:, :2]) < 0.1)
        reward_ang = (torch.abs(self.base_ang_vel[:, -1]) * (torch.abs(self.commands[:, 2]) < 0.1)).unsqueeze(dim=-1)
        return torch.sum(torch.cat((reward_lin, reward_ang), dim=-1), dim=-1)
    
    def _reward_feet_vel_still(self):
        # Penalize displacement and rotation at zero commands
        feet_vel_l = quat_rotate_inverse(self.base_quat, self.feet_state[:,0, 7:10])
        feet_vel_r = quat_rotate_inverse(self.base_quat, self.feet_state[:,1, 7:10])
        # 将左右脚的速度值在 x 和 y 方向分别拼接
        foot_vel_x = torch.stack((feet_vel_l[:, 0], feet_vel_r[:, 0]), dim=1)
        foot_vel_y = torch.stack((feet_vel_l[:, 1], feet_vel_r[:, 1]), dim=1)
        # left_foot_vel = feet_xy_vel[:, 0, :]  # 左脚的 xy 速度，形状 [num_envs, 2]
        # right_foot_vel = feet_xy_vel[:, 1, :]  # 右脚的 xy 速度，形状 [num_envs, 2]     
        reward_l = torch.norm(foot_vel_x, dim=1) * (torch.abs(self.commands[:, :1]) < 0.1).float()
        reward_r = torch.norm(foot_vel_y, dim=1) * (torch.abs(self.commands[:, 1:2]) < 0.1).float()
        return torch.sum(torch.cat((reward_l, reward_r), dim=-1), dim=-1)
    
    def _reward_feet_alignment_x(self):
        # 获取相对于机体坐标系的左右足端位置
        base_position = self.root_states[:, 0:3]
        feet_pos_l = quat_rotate_inverse(self.base_quat, self.feet_state[:, 0, 0:3] - base_position)
        feet_pos_r = quat_rotate_inverse(self.base_quat, self.feet_state[:, 1, 0:3] - base_position)

        # 提取左右足端在 x 方向上的位置
        feet_pos_lx = feet_pos_l[:, 0]
        feet_pos_rx = feet_pos_r[:, 0]

        # 判断左右脚是否在同一侧
        same_direction = torch.sign(feet_pos_lx) * torch.sign(feet_pos_rx) > 0

        # 计算两只脚 x 位置的差值
        feet_x_diff = torch.abs(feet_pos_lx - feet_pos_rx)

        feet_xzeors_diff = torch.abs(feet_pos_lx + feet_pos_rx)
        final_xzeors_reward = feet_xzeors_diff* (torch.abs(self.base_lin_vel[:, 1]) < 0.1).float()

        # 计算奖励和惩罚
        reward = torch.zeros_like(feet_pos_lx)
        reward[same_direction] = feet_x_diff[same_direction]  # 奖励同侧且位置接近
        reward[~same_direction] = feet_x_diff[~same_direction] * 2  # 惩罚一前一后

        # 获取机器人在 x 方向上的线速度
        linear_velocity_x = self.base_lin_vel[:, 0]#self.base_lin_vel[:, 0]
        velocity_threshold = 0.1 # 速度阈值

        # 判断机器人是否处于低速状态
        low_velocity = torch.abs(linear_velocity_x) < velocity_threshold

        # 在低速情况下应用奖励函数
        final_x_reward = reward * low_velocity.float()

        return final_x_reward #+ final_xzeors_reward
    
    def _reward_feet_alignment_y(self):
        # 获取相对于机体坐标系的左右足端位置
        base_position = self.root_states[:, 0:3]
        feet_pos_l = quat_rotate_inverse(self.base_quat, self.feet_state[:, 0, 0:3] - base_position)
        feet_pos_r = quat_rotate_inverse(self.base_quat, self.feet_state[:, 1, 0:3] - base_position)

        feet_pos_ly = feet_pos_l[:, 1]
        feet_pos_ry = feet_pos_r[:, 1]


        feet_y_diff = torch.abs(feet_pos_ly + feet_pos_ry)
        final_y_reward = feet_y_diff* (torch.abs(self.base_lin_vel[:, 1]) < 0.1).float()


        return final_y_reward
    
    def _reward_above_vel_range(self):
        reward_x = torch.max(self.base_lin_vel[:, 0] - self.command_ranges["lin_vel_x"][1], torch.zeros_like(self.base_lin_vel[:, 0])) +\
                torch.max(-self.base_lin_vel[:, 0] + self.command_ranges["lin_vel_x"][0], torch.zeros_like(self.base_lin_vel[:, 0]))
        
        reward_y = torch.max(self.base_lin_vel[:, 1] - self.command_ranges["lin_vel_y"][1], torch.zeros_like(self.base_lin_vel[:, 1])) +\
                torch.max(-self.base_lin_vel[:, 1] + self.command_ranges["lin_vel_y"][0], torch.zeros_like(self.base_lin_vel[:, 1]))
        
        return reward_x + reward_y

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_distance(self):
        reward = 0
        for i in range(self.feet_state.shape[1] - 1):
            for j in range(i + 1, self.feet_state.shape[1]):
                feet_distance = torch.norm(
                    self.feet_state[:, i, :2] - self.feet_state[:, j, :2], dim=-1
                )
            reward += torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        return (~self.reset_buf).float() * self.dt

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        return term_2
    
    def _reward_feet_y_force(self):
        """
        Penalize y-direction ground reaction forces in local frame when feet are in contact with ground.
        This encourages minimizing lateral forces during stance phase.
        """
        
        # Transform forces to base frame
        forces_local_r = quat_rotate_inverse(self.base_quat, self.contact_forces[:, 0, 0:3])
        forces_local_l = quat_rotate_inverse(self.base_quat, self.contact_forces[:, 1, 0:3])
        
        # Get y-direction forces in local frame
        foot_forces_y_r = forces_local_r[:, 1] # y-direction force in local frame
        foot_forces_y_l = forces_local_l[:, 1] # y-direction force in local frame
        
        # Only consider y-forces when foot is in contact
        y_forces_r = torch.abs(foot_forces_y_r) * self.contact_filt[:, 0].float()
        y_forces_l = torch.abs(foot_forces_y_l) * self.contact_filt[:, 1].float()
        
        return y_forces_r + y_forces_l
    
    def _reward_feet_contact_vel(self):
        """
        Penalize z-direction velocities when feet are close to the ground.
        This encourages smooth landing and prevents high-impact foot strikes.
        """
        
        # Define height threshold for velocity penalty (e.g. 0.1m above ground)
        height_threshold = 0.005 + 0.03
        
        # Get z velocities in world frame
        feet_vel = self.feet_state[:, :, 7:10] # Index 9 is z-velocity
        
        # Only penalize z velocities when feet are below threshold
        close_to_ground = self.feet_height < height_threshold
        penalized_vel = torch.norm(feet_vel, dim=-1) * close_to_ground.float()
        
        # Sum penalties across all feet
        return torch.sum(penalized_vel, dim=1)
    
    
