"""
renderer.py
===========

Vulkan-backed offscreen point renderer for the 3DGS alignment engine.

Design constraints (Milestone 1):
    * Target hardware has NO discrete GPU.  We *explicitly* prefer the
      integrated GPU, then a software rasteriser (Mesa lavapipe / LLVMpipe),
      and only fall back to a discrete GPU when nothing else is available.
    * Points are drawn as lightweight Vulkan point primitives
      (VK_PRIMITIVE_TOPOLOGY_POINT_LIST — Vulkan's equivalent of `gl_Points`).
    * The 4×4 alignment transform is applied to the **scene root**, never
      baked into the vertex data, so a UI button can refresh the rendered
      view in a single push-constant update.
    * If Vulkan initialisation fails (driver missing, surface mismatch, …)
      we fall back to a NumPy software rasteriser so the app still launches.

Public surface (used by app.py):

    r = VulkanRenderer(width=900, height=700)
    r.set_points(positions, colors)              # (N,3) float32 each
    r.set_scene_root_transform(M4x4)             # numpy (4,4)
    r.set_view(camera_distance=2.5, yaw=…, pitch=…)
    rgba = r.render()                            # (H, W, 4) uint8

The renderer is single-threaded; it owns its own Vulkan device and is
intended to be driven from the GUI thread.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Vulkan wrapper is optional at import time so loader.py / alignment.py work
# in headless environments without Vulkan loader libs (e.g. CI containers).
try:
    import vulkan as vk
    _VULKAN_AVAILABLE = True
except Exception as _e:                           # pragma: no cover
    vk = None                                     # type: ignore
    _VULKAN_AVAILABLE = False
    _VULKAN_IMPORT_ERROR = _e


SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 700
CLEAR_COLOR = (0.08, 0.09, 0.11, 1.0)             # dark slate background


# ---------------------------------------------------------------------------
# Small matrix utilities (column-major Vulkan/GLSL convention)
# ---------------------------------------------------------------------------

def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Right-handed perspective with Vulkan's [0,1] depth range and flipped Y."""
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f                                  # Vulkan Y is flipped vs GL
    m[2, 2] = zfar / (znear - zfar)
    m[2, 3] = (zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


# ---------------------------------------------------------------------------
# Camera (orbit) — owned by the renderer, driven by the GUI
# ---------------------------------------------------------------------------

@dataclass
class OrbitCamera:
    target: np.ndarray = None             # type: ignore[assignment]
    distance: float = 3.0
    yaw_deg: float = 35.0
    pitch_deg: float = 22.0
    fovy_deg: float = 45.0

    def __post_init__(self):
        if self.target is None:
            self.target = np.zeros(3, dtype=np.float32)

    def eye(self) -> np.ndarray:
        ya = math.radians(self.yaw_deg)
        pa = math.radians(self.pitch_deg)
        cp = math.cos(pa)
        e = np.array([
            self.distance * cp * math.sin(ya),
            self.distance * math.sin(pa),
            self.distance * cp * math.cos(ya),
        ], dtype=np.float32) + self.target
        return e

    def view(self) -> np.ndarray:
        return look_at(self.eye(), self.target,
                       np.array([0, 1, 0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class VulkanRenderer:
    """
    Offscreen Vulkan point renderer with a NumPy software fallback.

    The fallback is exposed at `r.is_vulkan` (False if Vulkan failed at init).
    Either way `r.render()` returns an RGBA uint8 image of shape (H, W, 4).
    """

    def __init__(self,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 point_size: float = 2.0,
                 prefer_integrated: bool = True):
        self.width = int(width)
        self.height = int(height)
        self.point_size = float(point_size)
        self.prefer_integrated = bool(prefer_integrated)

        # Scene state — kept on host, uploaded each frame as a push constant.
        self.scene_root = np.eye(4, dtype=np.float32)
        self.camera = OrbitCamera()
        self._positions: Optional[np.ndarray] = None    # (N,3) float32
        self._colors: Optional[np.ndarray] = None       # (N,3) float32
        self._n_points = 0
        self._verts_dirty = True

        self.is_vulkan = False
        self.device_name = "software-fallback"
        self.device_type = "cpu"
        self._vk_init_error: Optional[str] = None

        if _VULKAN_AVAILABLE:
            try:
                self._init_vulkan()
                self.is_vulkan = True
            except Exception as e:                # noqa: BLE001
                self._vk_init_error = repr(e)
                self._teardown_vulkan_safe()

    # ----- public scene-state API -------------------------------------------

    def set_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3); got {positions.shape}")
        if colors.shape != positions.shape:
            raise ValueError("positions and colors must have matching shape")
        self._positions = positions.astype(np.float32, copy=False)
        self._colors = colors.astype(np.float32, copy=False)
        self._n_points = int(positions.shape[0])
        self._verts_dirty = True

        # Centre the camera on the new cloud's mid-point at a sensible distance.
        if self._n_points > 0:
            mn = self._positions.min(axis=0)
            mx = self._positions.max(axis=0)
            self.camera.target = ((mn + mx) * 0.5).astype(np.float32)
            self.camera.distance = float(np.linalg.norm(mx - mn) * 1.05 + 1e-3)

    def set_scene_root_transform(self, m4: np.ndarray) -> None:
        if m4.shape != (4, 4):
            raise ValueError("transform must be (4,4)")
        self.scene_root = m4.astype(np.float32, copy=True)

    def set_view(self, yaw_deg: Optional[float] = None,
                 pitch_deg: Optional[float] = None,
                 distance: Optional[float] = None) -> None:
        if yaw_deg is not None:
            self.camera.yaw_deg = float(yaw_deg)
        if pitch_deg is not None:
            self.camera.pitch_deg = float(pitch_deg)
        if distance is not None:
            self.camera.distance = float(distance)

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width = int(width)
        self.height = int(height)
        if self.is_vulkan:
            self._recreate_targets()

    # ----- rendering -------------------------------------------------------

    def render(self) -> np.ndarray:
        """Render the current scene and return an (H,W,4) uint8 RGBA image."""
        if self.is_vulkan:
            try:
                return self._render_vulkan()
            except Exception as e:                # noqa: BLE001
                self._vk_init_error = f"render error: {e!r}"
                self.is_vulkan = False
                self._teardown_vulkan_safe()
        return self._render_software()

    def mvp(self) -> np.ndarray:
        """projection · view · sceneRoot — the same matrix the shader uses."""
        aspect = max(self.width / max(self.height, 1), 1e-3)
        p = perspective(self.camera.fovy_deg, aspect, 0.05, 1000.0)
        v = self.camera.view()
        return (p @ v @ self.scene_root).astype(np.float32)

    # ----- diagnostics -----------------------------------------------------

    def info(self) -> dict:
        return {
            "is_vulkan": self.is_vulkan,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "init_error": self._vk_init_error,
            "width": self.width, "height": self.height,
            "points": self._n_points,
            "point_size": self.point_size,
        }

    def close(self) -> None:
        # Explicit destruction is opt-in: the python `vulkan` wrapper's
        # cffi handles can race with module-finalisation at interpreter exit
        # (segfault inside libvulkan).  Apps that want clean destruction can
        # call close() while the interpreter is still fully alive.
        self._teardown_vulkan_safe()

    # NOTE: deliberately no __del__.  See close() for the rationale.

    # ======================================================================
    # Software fallback rasteriser
    # ======================================================================

    def _render_software(self) -> np.ndarray:
        H, W = self.height, self.width
        img = np.empty((H, W, 4), dtype=np.uint8)
        img[..., 0] = int(CLEAR_COLOR[0] * 255)
        img[..., 1] = int(CLEAR_COLOR[1] * 255)
        img[..., 2] = int(CLEAR_COLOR[2] * 255)
        img[..., 3] = 255

        if self._positions is None or self._n_points == 0:
            return img

        mvp = self.mvp()                                  # (4,4)
        n = self._n_points
        pts = self._positions
        cols = self._colors
        # Transform to clip space.
        ones = np.ones((n, 1), dtype=np.float32)
        clip = np.concatenate([pts, ones], axis=1) @ mvp.T  # (N,4)
        w = clip[:, 3:4]
        # Cull behind-camera and degenerate w.
        valid = (w[:, 0] > 1e-4)
        if not np.any(valid):
            return img
        ndc = clip[valid] / w[valid]
        cols_v = cols[valid]

        # Depth cull to [0,1] (Vulkan) and screen rect.
        z = ndc[:, 2]
        in_z = (z >= 0.0) & (z <= 1.0)
        ndc = ndc[in_z]
        cols_v = cols_v[in_z]
        if ndc.shape[0] == 0:
            return img

        x = ((ndc[:, 0] * 0.5 + 0.5) * W).astype(np.int32)
        y = ((ndc[:, 1] * 0.5 + 0.5) * H).astype(np.int32)
        in_screen = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x = x[in_screen]
        y = y[in_screen]
        cols_v = cols_v[in_screen]
        if x.size == 0:
            return img

        # Sort back-to-front so nearer points overwrite farther ones.
        z_in = ndc[in_z][in_screen, 2]
        order = np.argsort(-z_in)        # descending z (farther first)
        x = x[order]; y = y[order]; cols_v = cols_v[order]

        rgb = np.clip(cols_v, 0, 1)
        img[y, x, 0] = (rgb[:, 0] * 255).astype(np.uint8)
        img[y, x, 1] = (rgb[:, 1] * 255).astype(np.uint8)
        img[y, x, 2] = (rgb[:, 2] * 255).astype(np.uint8)
        img[y, x, 3] = 255
        return img

    # ======================================================================
    # Vulkan path
    # ======================================================================

    # ------ initialisation ------------------------------------------------

    def _init_vulkan(self) -> None:
        self._instance = None
        self._phys = None
        self._device = None
        self._queue = None
        self._queue_family = -1
        self._command_pool = None
        self._render_pass = None
        self._pipeline_layout = None
        self._pipeline = None
        self._color_img = None
        self._color_mem = None
        self._color_view = None
        self._depth_img = None
        self._depth_mem = None
        self._depth_view = None
        self._framebuffer = None
        self._vertex_buf = None
        self._vertex_mem = None
        self._vertex_cap = 0
        self._readback_buf = None
        self._readback_mem = None
        self._fence = None
        self._cmd_buf = None
        self._vshader = None
        self._fshader = None

        self._create_instance()
        self._pick_physical_device()
        self._create_logical_device()
        self._create_shader_modules()
        self._create_targets()
        self._create_render_pass()
        self._create_framebuffer()
        self._create_pipeline()
        self._create_command_pool_and_buffer()
        self._fence = vk.vkCreateFence(
            self._device,
            vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                 flags=0),
            None,
        )

    def _create_instance(self) -> None:
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="3dgs-aligner-m1",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="m1-renderer",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app,
            enabledLayerCount=0,
            enabledExtensionCount=0,
        )
        self._instance = vk.vkCreateInstance(info, None)

    def _pick_physical_device(self) -> None:
        devs = vk.vkEnumeratePhysicalDevices(self._instance)
        if not devs:
            raise RuntimeError("no Vulkan physical devices")

        # Score-based selection.  Higher == preferred.
        # On purpose: discrete GPU is *least* preferred per project constraint.
        type_score = {
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 100,
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: 80,         # llvmpipe/lavapipe
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 60,
            vk.VK_PHYSICAL_DEVICE_TYPE_OTHER: 40,
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                30 if self.prefer_integrated else 150,
        }

        best = None
        best_score = -1
        for d in devs:
            p = vk.vkGetPhysicalDeviceProperties(d)
            s = type_score.get(p.deviceType, 0)
            if s > best_score:
                best_score = s
                best = (d, p)

        if best is None:
            raise RuntimeError("no usable Vulkan device")
        self._phys, props = best
        self.device_name = str(props.deviceName)
        self.device_type = {
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "integrated",
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "discrete",
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: "software",
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "virtual",
        }.get(props.deviceType, "other")

        # Pick a queue family that supports graphics.
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self._phys)
        for i, f in enumerate(families):
            if f.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                self._queue_family = i
                break
        if self._queue_family < 0:
            raise RuntimeError("no graphics-capable queue family")

    def _create_logical_device(self) -> None:
        prio = [1.0]
        qinfo = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self._queue_family,
            queueCount=1,
            pQueuePriorities=prio,
        )
        feats = vk.VkPhysicalDeviceFeatures(
            largePoints=vk.VK_TRUE,           # only honoured if HW supports it
        )
        info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[qinfo],
            enabledLayerCount=0,
            enabledExtensionCount=0,
            pEnabledFeatures=feats,
        )
        self._device = vk.vkCreateDevice(self._phys, info, None)
        self._queue = vk.vkGetDeviceQueue(self._device, self._queue_family, 0)

    def _read_spv(self, name: str) -> bytes:
        path = os.path.join(SHADER_DIR, name)
        with open(path, "rb") as f:
            return f.read()

    def _create_shader_modules(self) -> None:
        v = self._read_spv("points.vert.spv")
        f = self._read_spv("points.frag.spv")
        self._vshader = vk.vkCreateShaderModule(
            self._device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(v), pCode=v,
            ),
            None,
        )
        self._fshader = vk.vkCreateShaderModule(
            self._device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(f), pCode=f,
            ),
            None,
        )

    def _find_memory_type(self, type_bits: int, props: int) -> int:
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self._phys)
        for i in range(mem_props.memoryTypeCount):
            mt = mem_props.memoryTypes[i]
            if (type_bits & (1 << i)) and (mt.propertyFlags & props) == props:
                return i
        raise RuntimeError("no suitable Vulkan memory type")

    def _create_image(self, width: int, height: int,
                      fmt: int, usage: int, aspect: int):
        info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=fmt,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        img = vk.vkCreateImage(self._device, info, None)
        req = vk.vkGetImageMemoryRequirements(self._device, img)
        mem = vk.vkAllocateMemory(
            self._device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=req.size,
                memoryTypeIndex=self._find_memory_type(
                    req.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ),
            None,
        )
        vk.vkBindImageMemory(self._device, img, mem, 0)
        view = vk.vkCreateImageView(
            self._device,
            vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=fmt,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=aspect,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            ),
            None,
        )
        return img, mem, view

    def _create_targets(self) -> None:
        self._color_img, self._color_mem, self._color_view = self._create_image(
            self.width, self.height,
            vk.VK_FORMAT_R8G8B8A8_UNORM,
            vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            vk.VK_IMAGE_ASPECT_COLOR_BIT,
        )
        self._depth_img, self._depth_mem, self._depth_view = self._create_image(
            self.width, self.height,
            vk.VK_FORMAT_D32_SFLOAT,
            vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            vk.VK_IMAGE_ASPECT_DEPTH_BIT,
        )
        self._readback_size = self.width * self.height * 4
        self._readback_buf, self._readback_mem = self._create_buffer(
            self._readback_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def _create_render_pass(self) -> None:
        color_attach = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        )
        depth_attach = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_D32_SFLOAT,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        depth_ref = vk.VkAttachmentReference(
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_ref],
            pDepthStencilAttachment=depth_ref,
        )
        dep = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL, dstSubpass=0,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )
        info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=2,
            pAttachments=[color_attach, depth_attach],
            subpassCount=1, pSubpasses=[subpass],
            dependencyCount=1, pDependencies=[dep],
        )
        self._render_pass = vk.vkCreateRenderPass(self._device, info, None)

    def _create_framebuffer(self) -> None:
        attachments = [self._color_view, self._depth_view]
        info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self._render_pass,
            attachmentCount=2, pAttachments=attachments,
            width=self.width, height=self.height, layers=1,
        )
        self._framebuffer = vk.vkCreateFramebuffer(self._device, info, None)

    def _create_pipeline(self) -> None:
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=self._vshader, pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self._fshader, pName="main",
            ),
        ]
        binding = vk.VkVertexInputBindingDescription(
            binding=0, stride=6 * 4,           # 6 floats = pos3 + col3
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        )
        attrs = [
            vk.VkVertexInputAttributeDescription(
                location=0, binding=0,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT, offset=0,
            ),
            vk.VkVertexInputAttributeDescription(
                location=1, binding=0,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT, offset=12,
            ),
        ]
        vinput = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1, pVertexBindingDescriptions=[binding],
            vertexAttributeDescriptionCount=2, pVertexAttributeDescriptions=attrs,
        )
        iasm = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport = vk.VkViewport(
            x=0.0, y=0.0, width=float(self.width), height=float(self.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=vk.VkExtent2D(width=self.width, height=self.height),
        )
        vps = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1, pViewports=[viewport],
            scissorCount=1, pScissors=[scissor],
        )
        rs = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            lineWidth=1.0,
        )
        ms = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=vk.VK_FALSE,
        )
        ds = vk.VkPipelineDepthStencilStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_TRUE,
            depthCompareOp=vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            depthBoundsTestEnable=vk.VK_FALSE,
            stencilTestEnable=vk.VK_FALSE,
        )
        ca = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT |
                             vk.VK_COLOR_COMPONENT_G_BIT |
                             vk.VK_COLOR_COMPONENT_B_BIT |
                             vk.VK_COLOR_COMPONENT_A_BIT),
            blendEnable=vk.VK_FALSE,
        )
        cb = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1, pAttachments=[ca],
        )
        push = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            offset=0, size=4 * 16 + 4,                  # mat4 + float pointSize
        )
        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=0,
            pushConstantRangeCount=1, pPushConstantRanges=[push],
        )
        self._pipeline_layout = vk.vkCreatePipelineLayout(self._device, layout_info, None)
        pinfo = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2, pStages=stages,
            pVertexInputState=vinput,
            pInputAssemblyState=iasm,
            pViewportState=vps,
            pRasterizationState=rs,
            pMultisampleState=ms,
            pDepthStencilState=ds,
            pColorBlendState=cb,
            layout=self._pipeline_layout,
            renderPass=self._render_pass,
            subpass=0,
        )
        pipelines = vk.vkCreateGraphicsPipelines(
            self._device, vk.VK_NULL_HANDLE, 1, [pinfo], None,
        )
        self._pipeline = pipelines[0]

    def _create_buffer(self, size: int, usage: int, props: int):
        info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=int(size), usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self._device, info, None)
        req = vk.vkGetBufferMemoryRequirements(self._device, buf)
        mem = vk.vkAllocateMemory(
            self._device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=req.size,
                memoryTypeIndex=self._find_memory_type(req.memoryTypeBits, props),
            ),
            None,
        )
        vk.vkBindBufferMemory(self._device, buf, mem, 0)
        return buf, mem

    def _create_command_pool_and_buffer(self) -> None:
        self._command_pool = vk.vkCreateCommandPool(
            self._device,
            vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self._queue_family,
            ),
            None,
        )
        cbs = vk.vkAllocateCommandBuffers(
            self._device,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self._command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )
        self._cmd_buf = cbs[0]

    # ------ uploads -------------------------------------------------------

    def _ensure_vertex_buffer(self, n_points: int) -> None:
        needed = max(n_points, 1) * 24                # 6 floats × 4 bytes
        if self._vertex_buf is not None and needed <= self._vertex_cap:
            return
        if self._vertex_buf is not None:
            vk.vkDestroyBuffer(self._device, self._vertex_buf, None)
            vk.vkFreeMemory(self._device, self._vertex_mem, None)
            self._vertex_buf = None
            self._vertex_mem = None
        # Grow with slack to amortise reallocation.
        cap = max(needed * 2, 24 * 4096)
        self._vertex_buf, self._vertex_mem = self._create_buffer(
            cap,
            vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        self._vertex_cap = cap

    def _upload_vertices(self) -> None:
        if self._positions is None or self._n_points == 0:
            return
        n = self._n_points
        self._ensure_vertex_buffer(n)
        interleaved = np.empty((n, 6), dtype=np.float32)
        interleaved[:, 0:3] = self._positions
        interleaved[:, 3:6] = self._colors
        data = interleaved.tobytes()
        ptr = vk.vkMapMemory(self._device, self._vertex_mem, 0, len(data), 0)
        # ptr is a cffi cdata void* (not a raw int) — write via ffi.memmove.
        vk.ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self._device, self._vertex_mem)
        self._verts_dirty = False

    # ------ recreate on resize -------------------------------------------

    def _recreate_targets(self) -> None:
        vk.vkDeviceWaitIdle(self._device)
        for handle, destroyer in [
            (self._framebuffer, vk.vkDestroyFramebuffer),
            (self._color_view, vk.vkDestroyImageView),
            (self._depth_view, vk.vkDestroyImageView),
        ]:
            if handle is not None:
                destroyer(self._device, handle, None)
        for img, mem in [(self._color_img, self._color_mem),
                         (self._depth_img, self._depth_mem)]:
            if img is not None:
                vk.vkDestroyImage(self._device, img, None)
            if mem is not None:
                vk.vkFreeMemory(self._device, mem, None)
        if self._readback_buf is not None:
            vk.vkDestroyBuffer(self._device, self._readback_buf, None)
            vk.vkFreeMemory(self._device, self._readback_mem, None)
        self._create_targets()
        self._create_framebuffer()
        # Pipeline carries static viewport; rebuild it too.
        if self._pipeline is not None:
            vk.vkDestroyPipeline(self._device, self._pipeline, None)
            self._pipeline = None
        if self._pipeline_layout is not None:
            vk.vkDestroyPipelineLayout(self._device, self._pipeline_layout, None)
            self._pipeline_layout = None
        self._create_pipeline()

    # ------ frame ---------------------------------------------------------

    def _record_command_buffer(self) -> None:
        cb = self._cmd_buf
        vk.vkResetCommandBuffer(cb, 0)
        vk.vkBeginCommandBuffer(
            cb,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        clears = [
            vk.VkClearValue(color=vk.VkClearColorValue(float32=list(CLEAR_COLOR))),
            vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)),
        ]
        rp_begin = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self._render_pass,
            framebuffer=self._framebuffer,
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=self.width, height=self.height),
            ),
            clearValueCount=2, pClearValues=clears,
        )
        vk.vkCmdBeginRenderPass(cb, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)

        if self._n_points > 0 and self._vertex_buf is not None:
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline)
            vk.vkCmdBindVertexBuffers(cb, 0, 1, [self._vertex_buf], [0])

            mvp = self.mvp().astype(np.float32, copy=False)
            # GLSL mat4 is column-major; numpy is row-major → transpose first.
            push_bytes = mvp.T.tobytes() + struct.pack("f", self.point_size)
            # cffi can't deduce size of `void[]` from a python bytes object;
            # an explicit uint8 array makes the type unambiguous.
            push_arr = vk.ffi.new("uint8_t[]", push_bytes)
            vk.vkCmdPushConstants(
                cb, self._pipeline_layout,
                vk.VK_SHADER_STAGE_VERTEX_BIT,
                0, len(push_bytes), push_arr,
            )
            vk.vkCmdDraw(cb, self._n_points, 1, 0, 0)

        vk.vkCmdEndRenderPass(cb)

        # Copy the rendered colour image into the host-visible readback buffer.
        region = vk.VkBufferImageCopy(
            bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
        )
        vk.vkCmdCopyImageToBuffer(
            cb, self._color_img,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self._readback_buf, 1, [region],
        )
        vk.vkEndCommandBuffer(cb)

    def _render_vulkan(self) -> np.ndarray:
        if self._verts_dirty:
            self._upload_vertices()

        self._record_command_buffer()

        vk.vkResetFences(self._device, 1, [self._fence])
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1, pCommandBuffers=[self._cmd_buf],
        )
        vk.vkQueueSubmit(self._queue, 1, [submit], self._fence)
        vk.vkWaitForFences(self._device, 1, [self._fence], vk.VK_TRUE, 5_000_000_000)

        # vkMapMemory in this wrapper returns a cffi buffer of the right
        # length already, so we can feed it straight to np.frombuffer and
        # copy out before unmap.
        ptr = vk.vkMapMemory(self._device, self._readback_mem, 0,
                              self._readback_size, 0)
        arr = np.frombuffer(ptr, dtype=np.uint8, count=self._readback_size)
        arr = arr.reshape((self.height, self.width, 4)).copy()
        vk.vkUnmapMemory(self._device, self._readback_mem)
        return arr

    # ------ teardown ------------------------------------------------------

    def _teardown_vulkan_safe(self) -> None:
        if not _VULKAN_AVAILABLE or self._instance is None:
            return
        try:
            vk.vkDeviceWaitIdle(self._device)
        except Exception:
            pass
        destroyers: List[Tuple[object, object]] = [
            (self._fence, vk.vkDestroyFence),
            (self._pipeline, vk.vkDestroyPipeline),
            (self._pipeline_layout, vk.vkDestroyPipelineLayout),
            (self._framebuffer, vk.vkDestroyFramebuffer),
            (self._render_pass, vk.vkDestroyRenderPass),
            (self._color_view, vk.vkDestroyImageView),
            (self._depth_view, vk.vkDestroyImageView),
            (self._vshader, vk.vkDestroyShaderModule),
            (self._fshader, vk.vkDestroyShaderModule),
        ]
        for handle, fn in destroyers:
            if handle is not None and self._device is not None:
                try:
                    fn(self._device, handle, None)
                except Exception:
                    pass

        # Images / memory.
        for img, mem in [(self._color_img, self._color_mem),
                         (self._depth_img, self._depth_mem)]:
            if img is not None:
                try: vk.vkDestroyImage(self._device, img, None)
                except Exception: pass
            if mem is not None:
                try: vk.vkFreeMemory(self._device, mem, None)
                except Exception: pass

        # Buffers / memory.
        for buf, mem in [(self._vertex_buf, self._vertex_mem),
                         (self._readback_buf, self._readback_mem)]:
            if buf is not None:
                try: vk.vkDestroyBuffer(self._device, buf, None)
                except Exception: pass
            if mem is not None:
                try: vk.vkFreeMemory(self._device, mem, None)
                except Exception: pass

        if self._command_pool is not None and self._device is not None:
            try: vk.vkDestroyCommandPool(self._device, self._command_pool, None)
            except Exception: pass
        if self._device is not None:
            try: vk.vkDestroyDevice(self._device, None)
            except Exception: pass
        if self._instance is not None:
            try: vk.vkDestroyInstance(self._instance, None)
            except Exception: pass

        self._instance = None
        self._device = None
        self._queue = None
        self._command_pool = None
        self._render_pass = None
        self._pipeline = None
        self._pipeline_layout = None
        self._color_img = None
        self._color_mem = None
        self._color_view = None
        self._depth_img = None
        self._depth_mem = None
        self._depth_view = None
        self._framebuffer = None
        self._vertex_buf = None
        self._vertex_mem = None
        self._readback_buf = None
        self._readback_mem = None
        self._fence = None
        self._cmd_buf = None
        self._vshader = None
        self._fshader = None


# ---------------------------------------------------------------------------
# CLI: quick health check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="VulkanRenderer health check.")
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    args = ap.parse_args()

    r = VulkanRenderer(width=args.width, height=args.height)
    info = r.info()

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1024, 3)).astype(np.float32) * 0.4
    cols = rng.random((1024, 3)).astype(np.float32)
    r.set_points(pts, cols)
    img = r.render()
    info["render_image_shape"] = list(img.shape)
    info["render_image_nonzero"] = int(np.any(img[..., :3] > 30))
    r.close()
    print(json.dumps(info, indent=2))
