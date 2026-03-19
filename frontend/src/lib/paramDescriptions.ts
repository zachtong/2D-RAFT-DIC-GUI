/**
 * Human-readable parameter descriptions for tooltip display.
 *
 * Keys match parameter names used in ProcessingParams and StrainControls.
 * Descriptions are in Chinese for end-user display.
 */
export const PARAM_DESCRIPTIONS: Record<string, string> = {
  // --- Processing parameters ---
  memoryBudget:
    "每个瓦片的最大像素数（决定GPU显存用量）。\n" +
    "值越大 = 更少瓦片、更快处理，但可能导致显存不足 (OOM)。\n" +
    "建议使用模型推荐值。",
  blendQuality:
    "平铺块拼接质量。\n" +
    "Fast: 最少重叠、最快处理\n" +
    "Balanced: 适中重叠，推荐默认\n" +
    "High: 最大重叠，接缝最少但更慢",
  tileOverlap:
    "相邻瓦片之间的重叠像素数。\n" +
    "更大的重叠 = 更平滑的接缝过渡，但计算量增加。\n" +
    "推荐值: 32-128 px",
  roiMargin:
    "ROI边界外额外包含的像素数（上下文区域）。\n" +
    "为RAFT光流网络提供边界处的计算上下文。\n" +
    "推荐值: 64 px",
  smoothingRadius:
    "高斯平滑核的标准差（像素）。\n" +
    "值越大 = 位移场越平滑，但会丢失细节。\n" +
    "0 = 不平滑。推荐值: 0-3",
  mode:
    "处理模式:\n" +
    "• Accumulative: 每帧相对于第一帧计算位移（适合小变形）\n" +
    "• Incremental: 逐帧累加位移（适合大变形、裂纹扩展）",

  // --- Incremental mode parameters ---
  keyFrameMode:
    "参考帧更新策略:\n" +
    "• Every Frame: 每帧都更新参考帧\n" +
    "• Every N: 每N帧更新一次\n" +
    "• Custom: 手动指定关键帧",
  keyFrameInterval:
    "参考帧更新间隔（帧数）。\n" +
    "较小的间隔 = 更精确但误差累积更快。\n" +
    "推荐值: 5-20",
  maskSource:
    "裂纹掩码来源:\n" +
    "• Auto: 自动变形ROI掩码\n" +
    "• Custom: 从文件夹加载预制掩码",
  useMedianFilter:
    "对累积位移应用中值滤波。\n" +
    "减少增量模式下的误差累积，但会修改原始输出。",

  // --- Strain parameters ---
  strainMethod:
    "应变计算方法:\n" +
    "• Green-Lagrange: 考虑大变形旋转的严格公式\n" +
    "• Engineering: 简化的小变形近似",
  vsgSize:
    "虚拟应变计 (VSG) 的窗口大小（像素）。\n" +
    "必须为奇数。值越大 = 越平滑但空间分辨率越低。\n" +
    "推荐值: 9-51",
  step:
    "VSG的步长（像素）。\n" +
    "输出应变图的空间分辨率 = VSG / step。\n" +
    "值越大 = 计算越快但分辨率越低。",
  polyOrder:
    "VSG内拟合多项式的阶数。\n" +
    "1 = 线性（推荐），2 = 二次（更精确但需要更大VSG）",
  weighting:
    "VSG内数据点的权重分配:\n" +
    "• Gaussian: 中心权重高、边缘低（推荐）\n" +
    "• Uniform: 等权重",
  gaussianSigma:
    "VSG内高斯权重的标准差。\n" +
    "留空 = 自动（VSG/6）。较小值集中于中心。",
  spatialSigma:
    "应变计算前对位移场进行高斯平滑。\n" +
    "减少位移噪声对应变的影响。\n" +
    "0 = 不平滑。推荐值: 0-5 px",
  temporalSigma:
    "在时间方向对应变进行高斯平滑。\n" +
    "减少帧间应变波动。\n" +
    "0 = 不平滑。推荐值: 0-5 帧",
  condThreshold:
    "条件数阈值（数值稳定性过滤）。\n" +
    "VSG区域内条件数超过此值的点将被标记为NaN。\n" +
    "推荐值: 100-10000",
  erosion:
    "边界侵蚀距离（像素）。\n" +
    "移除ROI边界附近不可靠的应变值。\n" +
    "Auto = VSG/2",
};
