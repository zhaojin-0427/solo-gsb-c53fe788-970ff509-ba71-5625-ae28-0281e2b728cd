# 陶片接缝全局拼合服务 · Potsherd Global Seam Assembly

一个用 **Python + FastAPI + NumPy + OpenCV** 实现的 HTTP 服务。它根据成对边界点
为陶片估计**仅含旋转 + 平移**的相对位姿，并在**同一组全局位姿**下选出
**总分最高**的接缝集合——全局精确优化，**绝不逐边贪心**。

---

## 1. 它解决什么问题

每个候选接缝给出两片陶片边界上的若干匹配点对、一个正的接缝分数、位姿误差上限
（像素）与实体掩膜重叠像素上限。服务需要：

1. 由点对闭式估计相对刚体运动（Procrustes / Kabsch，二维，只允许正常旋转，
   不允许反射）；
2. 决定**采用哪些接缝**，使得存在**一组统一的全局片段位姿**，令所有被采用
   接缝同时满足：
   - 每个边界点对在全局位姿下的残差 ≤ 该接缝 `pose_error_limit`；
   - 闭环误差（接缝自带的点对拟合位姿与全局位姿推导的相对位姿之差）≤ 上限；
   - 两片陶片 alpha 实体掩膜在该相对位姿下的重叠像素数 ≤ `overlap_limit`；
3. 在所有可行集合中取**分数总和最大**者；总分相同时，取
   **已选接缝 ID 升序列表字典序最小**的集合；
4. 输出每个**连通拼合簇**的片段位姿、采用与淘汰接缝（含原因），并把该簇按
   **同一位姿**渲染成一张透明 PNG（Base64 返回）；**孤立片段各自成簇**。

不合格的接缝**不会参与成图**。

> **为什么不是逐边贪心？** 边之间通过共享片段位姿相互耦合：采用某条接缝会
> 改变闭环与所有相邻接缝的残差。本服务用**分支限界（branch & bound）在接缝
> 子集空间上做精确全局搜索**：目标是子集总分之最大，可行性由“是否存在一组
> 全局位姿”这一全局条件判定。搜索带有两项**可靠（不会改变最优解）**的加速：
> 候选图连通分量分别求解；搜索树上的每个节点用一个**单调性的位姿 LP 松弛**
> （12 边形外逼近 + 位姿修正一阶线性化）提前剪枝——LP 不可行则任何超集都不
> 可行；只有叶节点才做精确的 Gauss–Newton 全局位姿求解与掩膜光栅化校验。

---

## 2. 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听在 http://localhost:8000 ：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查，返回 `{"status":"ok"}` |
| POST | `/assemblies` | 提交陶片与候选接缝，返回全局拼合结果 |
| GET  | `/docs` | 交互式 OpenAPI 文档（可直接发请求） |

停止：

```bash
docker compose down
```

### 不用 Docker 的本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

跑测试：

```bash
pytest -q
```

---

## 3. 请求格式 `POST /assemblies`

```jsonc
{
  "fragments": [
    {
      "id": "shard-A",                       // 片段唯一 ID
      "image_base64": "iVBORw0KGgo...."       // 透明 PNG（alpha 通道），可带 data: 前缀
    }
  ],
  "seams": [
    {
      "id": "seam-AB",                        // 接缝唯一 ID
      "fragment_a": "shard-A",
      "fragment_b": "shard-B",
      "score": 12.5,                          // 正分数，越大越好
      "pose_error_limit": 3.0,                // 点对残差/闭环误差上限（像素）
      "overlap_limit": 200,                   // 实体掩膜重叠像素上限
      "pairs": [                              // 至少 2 对，且不能退化（共点）
        {"a": [39.0, 4.0], "b": [0.0, 4.0]},
        {"a": [39.0, 10.0], "b": [0.0, 10.0]}
      ]
    }
  ]
}
```

坐标为各片段**自身图像坐标系**中的像素坐标 `[x, y]`（原点左上角，x 向右、
y 向下）。点对 `a`、`b` 分别是边界点在 `fragment_a`、`fragment_b` 上的位置。

### 422（不可处理）情形

- 点对少于 2 对，或点对退化（全部重合，旋转不可观）；
- 片段 PNG 解码失败 / Base64 非法；
- 片段 alpha 实体掩膜为空；
- 接缝引用了不存在的片段、片段自连、ID 重复。

这些都返回 HTTP `422`，body 为 `{"detail": "...原因..."}`。

---

## 4. 实际请求与响应示例（多簇 + 闭环 + 淘汰）

仓库自带一个可直接生成真实请求并调用服务的脚本：

```bash
# 1) 生成 examples/request.json
#    5 张透明 PNG：A-B-C 一致闭环 + 独立接缝 D-E + 连接缝 E-A
python examples/make_example.py

# 2) 对正在运行的服务发起真实 POST，落地 examples/response.json 与渲染 PNG
python examples/make_example.py --call
```

示例候选图整体连通：`A-B-C-A` 是一条**一致闭环**（AB、BC 各平移 39px，
捷径 AC 平移 78px，三者共用一组全局位姿且残差/闭环误差为 0）；`D-E` 是
另一条独立接缝；`E-A` 把两组在候选图层面连起来，但它唯一满足点对的位姿
会让两片实体掩膜重叠 26 像素，而 `overlap_limit = 0`。因此**完整集合不可行**，
精确搜索在子集空间枚举——搜索过程中临时集合会同时包含“三片闭环簇”和
“尚未连入的独立接缝”两个连通分量——最终保留闭环（35 分）加独立接缝（8 分），
以 `mask_overlap_exceeds_limit` 淘汰连接缝，输出两个簇。

或用 curl：

```bash
curl -s -X POST http://localhost:8000/assemblies \
  -H 'Content-Type: application/json' \
  --data @examples/request.json | python -m json.tool
```

以下是对本机服务发起真实 POST 得到的响应（`examples/response.json`；
PNG 为真实 Base64，下方截断展示）：

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "fragment_ids": ["shard-A", "shard-B", "shard-C"],
      "poses": [
        {"fragment_id": "shard-A", "theta": 0.0, "tx": 0.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "shard-B", "theta": 0.0, "tx": 39.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "shard-C", "theta": 0.0, "tx": 78.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [
        {"id": "seam-AB", "fragment_a": "shard-A", "fragment_b": "shard-B",
         "score": 12.5, "pair_residual_max": 0.0, "loop_error": 0.0,
         "overlap_pixels": 26},
        {"id": "seam-AC", "fragment_a": "shard-A", "fragment_b": "shard-C",
         "score": 10.0, "pair_residual_max": 0.0, "loop_error": 0.0,
         "overlap_pixels": 0},
        {"id": "seam-BC", "fragment_a": "shard-B", "fragment_b": "shard-C",
         "score": 12.5, "pair_residual_max": 0.0, "loop_error": 0.0,
         "overlap_pixels": 26}
      ],
      "rejected_seams": [],
      "total_score": 35.0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAIUAAAA3CAYAAAAi2IlCAAAD...（透明 PNG）"
    },
    {
      "cluster_id": 2,
      "fragment_ids": ["shard-D", "shard-E"],
      "poses": [
        {"fragment_id": "shard-D", "theta": 0.0, "tx": 0.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "shard-E", "theta": 0.0, "tx": 39.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [
        {"id": "seam-DE", "fragment_a": "shard-D", "fragment_b": "shard-E",
         "score": 8.0, "pair_residual_max": 0.0, "loop_error": 0.0,
         "overlap_pixels": 26}
      ],
      "rejected_seams": [
        {"id": "seam-EA", "fragment_a": "shard-E", "fragment_b": "shard-A",
         "score": 6.0, "reason": "mask_overlap_exceeds_limit",
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 26}
      ],
      "total_score": 8.0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAF4AAAA3CAYAAACFKf38AAAC...（透明 PNG）"
    }
  ],
  "adopted_seam_ids": ["seam-AB", "seam-AC", "seam-BC", "seam-DE"],
  "rejected_seams": [
    {"id": "seam-EA", "reason": "mask_overlap_exceeds_limit",
     "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 26}
  ],
  "total_score": 43.0
}
```

`--call` 还会把每个簇按同一位姿渲染的透明 PNG 写到
`examples/cluster_1.png`、`examples/cluster_2.png`，可直接打开查看：
`cluster_1.png` 是三片陶片在统一全局位姿下拼回的闭环，`cluster_2.png`
是由独立接缝拼合的两片；被淘汰的连接缝**不会**把两组画到同一张图里。

> **多簇子集的松弛怎么算？** 搜索树节点上的临时接缝集合可能还是一片
> “森林”（例如闭环 + 一条尚未连入的接缝）。各连通分量不共享片段，
> 位姿修正变量彼此独立，因此松弛对**每个连通分量分别建一个 LP**
> （等价于在统一变量空间里求解一个块对角约束矩阵的 LP），全部可行
> 才算可行。这保证剪枝仍是可靠的必要性判据，绝不会误删可行集合。

### 淘汰接缝的原因码（`reason`）

| reason | 含义 |
|--------|------|
| `pairwise_pose_residual_exceeds_limit` | 该接缝点对自身的刚体拟合残差已超上限（全局位姿无法修复） |
| `pose_residual_exceeds_limit` | 与已采用接缝放在同一组全局位姿后，点对残差超上限 |
| `loop_closure_exceeds_limit` | 闭环误差（相对位姿不自洽）超上限 |
| `mask_overlap_exceeds_limit` | 该相对位姿下实体掩膜重叠像素数超上限 |
| `score_optimization_not_selected` | 本身可行，但为使总分最大而未被精确搜索选中 |

淘汰记录同时回填测得的 `pair_residual_max` / `loop_error` /
`overlap_pixels`（不适用时为 `null`）。淘汰接缝只出现在报告里，
**不会出现在任何簇的渲染图中**。

---

## 5. 模块结构

```
app/
  geometry.py    闭式刚体（旋转+平移）拟合、位姿复合/相对位姿
  posegraph.py   生成树初始化 + 全局 Gauss–Newton 位姿图求解（所有接缝同时拟合）
  relax.py       位姿可行性的 LP 松弛（按连通分量分簇建 LP，单调性、可靠剪枝）
  masks.py       alpha 掩膜、相对位姿下的实体重叠计数、透明 PNG 渲染
  assembler.py   解码/建模、可行性判定、分支限界精确全局选缝
  service.py     端到端编排：成簇、采用/淘汰归因、渲染与响应
  schemas.py     Pydantic 请求/响应模型
  main.py        FastAPI 应用（/health、/assemblies，422 处理）
examples/        真实请求/响应生成脚本与产物
tests/           API、几何、全局优化器（含对拍暴力枚举）与多片段闭环测试
```

### 位姿约定

片段位姿 `(theta, t)` 把片段局部坐标映到簇/世界坐标：
`p_world = R(theta) · p_local + t`。每个簇以该簇最小 ID 的片段为锚点
（位姿为恒等）。接缝内部保存由点对拟合的、把 `fragment_b` 局部坐标映到
`fragment_a` 局部坐标的相对位姿 `Rr = R_aᵀ R_b, tr = R_aᵀ(t_b − t_a)`。
