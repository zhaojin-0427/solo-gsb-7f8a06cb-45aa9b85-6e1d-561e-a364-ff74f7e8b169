# 录音片段定位（Audio Clip Locator）

本地网页工具：上传一段**参考录音**和一段**待找片段**，在参考录音中定位片段出现的位置，
在波形上标出命中区间，并可从命中处试听；唯一命中时还能带上下文导出精确裁剪的命中片段 WAV。

同一页面下方还提供**片段顺序核验**模块：上传参考录音与按**预期顺序**排列的 2～5 段片段，
判断它们是否能在参考录音中各取一处、按该顺序、互不重叠地出现，
并在参考波形上逐段标出完整链与相邻空档。

- 后端：Python + FastAPI + NumPy + SciPy
- 前端：原生 HTML / CSS / JavaScript（无任何前端依赖，波形用 Canvas 绘制）
- 匹配算法：归一化互相关（NCC），用 FFT 卷积一次算出所有候选起点，**不做逐位置逐采样点遍历**

## 一条命令启动

```bash
docker compose up --build
```

启动后在浏览器访问：**http://localhost:8000**

（端口映射为 `8000:8000`，如需修改请编辑 `docker-compose.yml` 中的 `ports`。）

也可以不用 Compose，纯 Docker：

```bash
docker build -t audio-locator .
docker run --rm -p 8000:8000 audio-locator
```

本地开发（不用 Docker）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000
```

## 输入要求

| 项目 | 参考录音 | 待找片段 |
| --- | --- | --- |
| 时长 | 1 ～ 30 秒 | 0.2 ～ 5 秒（且须短于参考录音） |
| 格式 | 未压缩 PCM WAV（16 位） | 同左 |
| 声道 | 单声道或双声道 | 同左 |
| 采样率 | 8 ～ 48 kHz | 8 ～ 48 kHz，**可与参考录音不同** |

片段应来自参考录音，但允许**不同的采样率**和**不同的整体音量**。
格式或时长不合要求时，接口返回 400 并在页面上给出具体原因
（非 WAV、非 16 位、声道数、采样率越界、时长越界等分别提示）。

## 处理流程

1. **校验与解码**：检查 WAV 头（PCM、16 位、声道数、采样率）与时长。
2. **转单声道**：双声道取左右均值。
3. **统一重采样**：用 `scipy.signal.resample_poly`（多相滤波，抗混叠）把两路信号
   重采样到两者中**较低**的采样率——片段中不存在的高频无法参与匹配，
   向高采样率对齐只会引入失配能量。
4. **去直流偏移**：两路信号各自减去均值。
5. **归一化互相关（FFT 加速）**：
   - 分子 `ref ⋆ clip` 与分母所需的滑动窗口能量均用 `scipy.signal.fftconvolve`
     一次算完，复杂度 O(N log N)，不逐位置遍历；
   - 每个候选起点得到 [-1, 1] 的相关系数，对整体音量变化天然不变。
6. **判定**：
   - 任一路信号峰值低于约 -60 dBFS → 判定**近乎静音，无法定位**；
   - 最高得分 < 0.8 → **无法定位**（返回实际最高分）；
   - 存在 ≥ 2 个起点，彼此相距 ≥ 0.25 秒、得分均 ≥ 0.8 且与最高分相差 ≤ 0.02
     → **定位不唯一**，列出全部候选；
   - 否则 → **唯一命中**，给出起止时间（毫秒精度）与得分。

## 页面功能

- 唯一命中：显示起止时间（`mm:ss.mmm`）与相关得分；
- 导出命中片段：唯一命中后可设置命中点**前后各 0～2 秒**上下文（步长 100 ms），
  服务端按**原始参考 WAV 的采样率、声道与 16 位 PCM 精确裁剪**（直接切 PCM、
  不重采样、不重新编码），超出参考录音边界时自动收紧，并在页面上显示**实际起止毫秒**，
  导出的 WAV 直接触发浏览器下载；定位不唯一、无法定位或范围无效时不生成文件，
  原定位提示保留；
- 定位不唯一：列出所有候选区间与得分，每个候选都可单独试听；
- 无法定位 / 输入不合要求：显示明确原因；
- 参考波形上高亮命中（候选）区间，点击波形可跳转播放位置，
  支持「从命中处试听」「从头播放」，播放时显示播放头。

## 片段顺序核验

页面下方第二个面板。上传**一段参考录音**与按**预期顺序**排列的 **2～5 段**片段
（格式、时长、采样率约束与定位页完全一致，每段都须短于参考录音；允许各段采样率不同，
统一重采样到参考与所有片段中**最低**的采样率），点击「上传并核验顺序」。

处理流程沿用定位页的去直流与 FFT 归一化互相关，判定规则：

1. **候选保留**：每段保留所有得分 ≥ 0.8 的起点——按得分降序贪心选取、
   同段候选彼此相隔至少 0.25 秒（同一峰只留最高分一处）；
   与单段定位不同，**较低分候选也保留**，供多段全局链判断。
2. **完整链搜索**：从每段候选中各取一处，要求按**输入顺序**排列且**互不重叠**
   （允许首尾相接、空档为 0）；以**未舍入得分之和**最高者为结果。
   动态规划以「上一段终点 ≤ 本段起点」为转移、前缀最大值优化，不做笛卡尔积枚举。
3. **结论**：
   - 某段近乎静音或没有 ≥ 0.8 的起点，或不存在满足约束的完整链
     → **顺序不成立**（页面与接口给出具体原因：`silence` / `segment_not_found` / `overlap`）；
   - 存在完整链 → **顺序成立**，唯一链列出各段**起止毫秒**与**相邻空档（毫秒）**，
     并在参考波形上用不同颜色逐段标出（段次标注在区间左上角），可从第一段试听；
   - 另有一条**不同链**与最高总分之差 ≤ **0.02**（按未舍入得分比较）
     → 仍给出最优链，同时提示**多解**并列出另一条链（波形上以同色虚线标注）。

「添加一段 / 移除」按钮可在 2～5 段之间调整片段槽；原定位与命中片段导出功能不变。

## 接口

### `POST /api/locate`

`multipart/form-data`，字段：`reference`（参考录音 WAV）、`clip`（待找片段 WAV）。

成功（200）返回示例：

```json
{
  "status": "located",
  "sample_rate": 16000,
  "best_score": 0.9987,
  "candidates": [{ "start_s": 3.25, "end_s": 4.45, "score": 0.9987 }],
  "clip_duration_s": 1.2,
  "reference": { "duration_s": 10.0, "sample_rate": 44100, "channels": 2 },
  "clip": { "duration_s": 1.2, "sample_rate": 16000, "channels": 1 }
}
```

`status` 取值：`located`（唯一命中）/ `ambiguous`（定位不唯一，含多个候选）/
`not_found`（无法定位，`reason` 为 `silence` 或 `low_score`）。

输入不合要求返回 `400`，`detail` 为具体中文错误信息。

### `POST /api/export`

`multipart/form-data`，字段：

| 字段 | 说明 |
| --- | --- |
| `reference` | 参考录音 WAV（同 `/api/locate` 的输入约束） |
| `clip` | 待找片段 WAV |
| `pre_ms` | 命中点前上下文（毫秒），整数，`0～2000`，步长 `100`，默认 `0` |
| `post_ms` | 命中点后上下文（毫秒），整数，`0～2000`，步长 `100`，默认 `0` |

服务端会**重新执行一次定位**（接口无状态）：仅当结果为唯一命中（`located`）时，
才按原始参考 WAV 的采样率、声道、16 位 PCM 直接裁剪命中区间加上下文，
返回 `200` 与 `audio/wav` 附件（`Content-Disposition: attachment`），
并通过响应头回传收紧后的实际范围：

- `X-Export-Start-Ms`：实际起点（毫秒）
- `X-Export-End-Ms`：实际终点（毫秒）

返回示例（响应头节选）：

```
Content-Type: audio/wav
Content-Disposition: attachment; filename="hit_2750-5150ms.wav"
X-Export-Start-Ms: 2750
X-Export-End-Ms: 5150
```

错误情况：

- `pre_ms` / `post_ms` 越界或不是 100 的整数倍、文件不合要求 → `400`；
- 定位不唯一或无法定位 → `409`，`detail` 为具体原因，**不生成文件**。

### `POST /api/sequence`

片段顺序核验。`multipart/form-data`，字段：`reference`（参考录音 WAV）、
`clips`（2～5 个片段 WAV，**字段顺序即预期顺序**）。

成功（200）且核验唯一时返回示例（3 段）：

```json
{
  "status": "verified",
  "sample_rate": 16000,
  "total_score": 2.9995,
  "multiple_solutions": false,
  "message": "顺序成立：3 段按输入顺序排列且互不重叠，链总得分 2.9995，核验唯一",
  "chain": {
    "segments": [
      { "segment": 1, "start_ms": 1000, "end_ms": 1600, "start_s": 1.0, "end_s": 1.6, "score": 1.0 },
      { "segment": 2, "start_ms": 3500, "end_ms": 4000, "start_s": 3.5, "end_s": 4.0, "score": 0.9996 },
      { "segment": 3, "start_ms": 6000, "end_ms": 6700, "start_s": 6.0, "end_s": 6.7, "score": 0.9999 }
    ],
    "gaps_ms": [1900, 2000],
    "total_score": 2.9995
  },
  "segments_report": [ { "segment": 1, "silent": false, "best_score": 1.0, "candidates": [ ... ] } ],
  "reference": { "duration_s": 10.0, "sample_rate": 16000, "channels": 1 },
  "clips": [ { "duration_s": 0.6, "sample_rate": 16000, "channels": 1 } ]
}
```

- 多解时额外返回 `"multiple_solutions": true` 与 `alternative_chain`（结构同 `chain`），
  其 `total_score` 与最优链之差 ≤ 0.02；
- 顺序不成立时 `status` 为 `sequence_not_established`，`reason` 为
  `silence`（参考或某段近静音）、`segment_not_found`（某段无 ≥ 0.8 起点）或
  `overlap`（各段都有候选但无按序、互不重叠的完整链），并附中文 `message`；
  `segments_report` 仍逐段列出候选，便于页面在波形上展示人工判断。

错误情况：片段数不在 2～5 之间、任一段非 WAV / 非 16 位 / 声道或采样率越界 /
时长越界（每段 0.2～5 秒且短于参考）→ `400`，`detail` 含段次与具体中文原因。

另有 `GET /api/health` 健康检查。

## 目录结构

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app
│   ├── main.py            # FastAPI 入口：/api/locate、/api/export、/api/sequence、静态页面托管
│   ├── audio.py           # WAV 校验解码、重采样、FFT 归一化互相关定位、命中片段导出、片段顺序核验
│   └── static             # 原生前端（index.html / style.css / app.js）
└── tests
    └── selftest.py        # 本地自测（合成 WAV 验证各分支）
```

## 自测

```bash
pip install -r requirements.txt httpx
python3 tests/selftest.py
```

覆盖：异采样率/异音量/含直流偏移的唯一命中、低分无法定位、近静音、
定位不唯一（含 0.25 秒间隔边界）、全部格式与时长校验分支；
以及命中片段导出：位精确裁剪、实际起止毫秒、开头/结尾越界收紧、
双声道保持、定位不唯一/失败返回 409、上下文非法返回 400；
以及片段顺序核验：唯一链起止毫秒与相邻空档（含首尾相接空档 0、异采样率片段）、
逆序无完整链、某段低分 / 静音、0.02 内多解与另一条链、
片段数 1 / 6 及非法输入返回 400。
