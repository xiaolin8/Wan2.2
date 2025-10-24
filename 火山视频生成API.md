### 创建视频生成任务 API

#### 文生视频

```bash
curl -X POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-seedance-1-0-pro-250528",
    "content": [
        {
            "type": "text",
            "text": "多个镜头。一名侦探进入一间光线昏暗的房间。他检查桌上的线索，手里拿起桌上的某个物品。镜头转向他正在思索。 --ratio 16:9"
        }
    ]
}'
```

```bash
{
  "id": "cgt-2025******-****"
}
```

---

#### 图生视频-首帧

```bash
curl -X POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-seedance-1-0-pro-250528",
    "content": [
        {
            "type": "text",
            "text": "女孩抱着狐狸，女孩睁开眼，温柔地看向镜头，狐狸友善地抱着，镜头缓缓拉出，女孩的头发被风吹动  --ratio adaptive  --dur 5"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/i2v_foxrgirl.png"
            }
        }
    ]
}'
```

----

#### 图生视频-首尾帧

```bash
curl -X POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-seedance-1-0-pro-250528",
    "content": [
         {
            "type": "text",
            "text": "360度环绕运镜"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_first_frame.jpeg"
            },
            "role": "first_frame"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_last_frame.jpeg"
            },
            "role": "last_frame"
        }
    ]
}
```

----

#### 图生视频-base64编码

```bash
curl -X POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-seedance-1-0-lite-i2v-250428",
    "content": [
        {
            "type": "text",
            "text": "女孩抱着狐狸，女孩睁开眼，温柔地看向镜头，狐狸友善地抱着，镜头缓缓拉出，女孩的头发被风吹动  --ratio adaptive  --dur 5"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,aHR0******cG5n"
            }
        }
    ]
}'
```



### 查询视频生成任务 API

```bash
curl -X GET https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-2025**** \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY"
```

```bash
{
  "id": "cgt-2025******-****",
  "model": "doubao-seedance-1-0-pro-250528",
  "status": "succeeded",
  "content": {
    "video_url": "https://ark-content-generation-cn-beijing.tos-cn-beijing.volces.com/doubao-seedance-1-0-pro/****"
  },
  "seed": 10,
  "resolution": "720p",
  "duration": 5,
  "ratio": "16:9",
  "framespersecond": 24,
  "usage": {
    "completion_tokens": 108900,
    "total_tokens": 108900
  },
  "created_at": 1743414619,
  "updated_at": 1743414673
}
```

**status** string

任务状态，以及相关的信息：

- queued：排队中。

- running：任务运行中。

- cancelled：取消任务，取消状态24h自动删除（只支持排队中状态的任务被取消）。

- succeeded： 任务成功。

- failed：任务失败。

----

### 查询视频生成任务列表

```bash
curl -X GET https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks?page_size=3&filter.status=succeeded& \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY"
```

```bash
{
  "total": 3,
  "items": [
    {
      "id": "cgt-2025******-****",
      "model": "doubao-seedance-1-0-pro-250528",
      "status": "succeeded",
      "content": {
        "video_url": "https://ark-content-generation-cn-beijing.tos-cn-beijing.volces.com/doubao-seedance-1-0-pro/****.mp4?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTY****%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250331T095113Z&X-Tos-Expires=86400&X-Tos-Signature=***&X-Tos-SignedHeaders=host"
      },
      "seed": 10,
      "resolution": "720p",
      "duration": 5,
      "ratio": "16:9",
      "framespersecond": 24,
      "usage": {
        "completion_tokens": 108900,
        "total_tokens": 108900
      },
      "created_at": 1743414619,
      "updated_at": 1743414673
    },
    {
      "id": "cgt-2025******-****",
      "model": "doubao-seedance-1-0-pro-250528",
      "status": "succeeded",
      "content": {
        "video_url": "https://ark-content-generation-cn-beijing.tos-cn-beijing.volces.com/xxx"
      },
      "seed": 23,
      "resolution": "720p",
      "duration": 5,
      "ratio": "16:9",
      "framespersecond": 24,
      "usage": {
        "completion_tokens": 82280,
        "total_tokens": 82280
      },
      "created_at": 1743406900,
      "updated_at": 1743406940
    },
    {
      "id": "cgt-2025******-****",
      "model": "doubao-seedance-1-0-pro-250528",
      "status": "succeeded",
      "content": {
        "video_url": "https://ark-content-generation-cn-beijing.tos-cn-beijing.volces.com/xxx"
      },
      "seed": 4,
      "resolution": "720p",
      "duration": 5,
      "ratio": "16:9",
      "framespersecond": 24,
      "usage": {
        "completion_tokens": 82280,
        "total_tokens": 82280
      },
      "created_at": 1743406900,
      "updated_at": 1743406946
    }
  ]
}
```

----

### 取消或删除视频生成任务

```bash
curl -X DELETE https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/$ID \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY"
```

任务状态不同，调用DELETE接口，执行的操作有所不同，具体说明如下：

| 当前任务状态 | 支持DELETE操作          | 操作含义                                  | DELETE操作后任务状态 |
| ------------ | ----------------------- | ----------------------------------------- | -------------------- |
| queued       | 是                      | 任务取消排队，任务状态被变更为cancelled。 | cancelled            |
| running      | 不支持                  | -                                         | -                    |
| succeeded    | 是                      | 删除视频生成任务记录，后续将不支持查询。  | -                    |
| failed       | 是                      | 删除视频生成任务记录，后续将不支持查询。  | -                    |
| cancelled    | 不支持 24小时后自动删除 | -                                         | -                    |



### 参考

- https://www.volcengine.com/docs/82379/1520757