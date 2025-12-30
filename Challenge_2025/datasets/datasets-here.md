## 下载特定文件夹/任务数据参考指令
```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Initialize an empty Git repository
git init <repo_name>
cd <repo_name>

# Set the remote repository, eg: AgiBotWorld-Alpha https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
git remote add origin <repo_url>

# Enable sparse-checkout
git sparse-checkout init

# Specify the folders and files
git sparse-checkout set <folder_name> <file_name> ... 

# Pull the data
git pull origin main

```


## 官方数据集
- **官方数据集**: [Hugging Face - AgiBotWorldChallenge-2025](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025) 约3.1T包含simdata与realrobot data


- **Manipulation-SimData**: 用于挑战阶段1的仿真数据。
  - 下载链接：[Manipulation-SimData](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData)
- **Manipulation-RealRobotData**: 用于挑战阶段2的真实机器人数据。
  - 下载链接：[Manipulation-RealRobotData](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-RealRobot)

训练集包含从 AgiBot 世界数据集中精选的 10 个代表性任务、超过 30,000 条优质轨迹，验证集包含30个挑选的样本，测试集包含30个非公开样本。

### 数据集目录结构
```
DATASET_ROOT/
├── train/
│   ├── 367-648961-000/
│   │   ├── head_color.mp4
│   │   ├── head_extrinsic_params_aligned.json
│   │   ├── head_intrinsic_params.json
│   │   └── proprio_stats.h5
│   ├── 367-648961-001/
│   │   ├── head_color.mp4
│   │   ├── head_extrinsic_params_aligned.json
│   │   ├── head_intrinsic_params.json
│   │   └── proprio_stats.h5
│   ├── {task_id}-{episode_id}-{step_id}/
│   │   ├── head_color.mp4
│   │   ├── head_extrinsic_params_aligned.json
│   │   ├── head_intrinsic_params.json
│   │   └── proprio_stats.h5
│   └── ...
├── val/
│   ├── 367-649524-000/
│   │   ├── head_color.mp4
│   │   ├── head_extrinsic_params_aligned.json
│   │   ├── head_intrinsic_params.json
│   │   └── proprio_stats.h5
│   └── ...
└── test/
    ├── {task_id}-{episode_id}-{step_id}/
    │   ├── frame.png
    │   ├── head_color.mp4 (NOT disclosed to participants)
    │   ├── head_extrinsic_params_aligned.json
    │   ├── head_intrinsic_params.json
    │   └── proprio_stats.h5
    └── ...
```

## 额外数据集
- **额外数据集**: 如果需要，从Hugging Face下载AgibotWorld-Alpha数据集（或AgibotWorld-Beta数据集，较大）。
  - 下载链接：[AgibotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha)，大约8T的数据,约100万条轨迹数据。
  - 下载链接：[AgibotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

### 额外数据集目录结构
Alpha数据集与Beta数据集的目录结构相同，只是数据量不同，结构如下所示：
```
data
├── task_info
│   ├── task_327.json
│   ├── task_352.json
│   └── ...
├── observations
│   ├── 327 # This represents the task id.
│   │   ├── 648642 # This represents the episode id.
│   │   │   ├── depth # This is a folder containing depth information saved in PNG format.
│   │   │   ├── videos # This is a folder containing videos from all camera perspectives.
│   │   ├── 648649
│   │   │   └── ...
│   │   └── ...
│   ├── 352
│   │   ├── 648544
│   │   │   ├── depth
│   │   │   ├── videos
│   │   ├── 648564
│   │   │   └── ...
│   └── ...
├── parameters
│   ├── 327
│   │   ├── 648642
│   │   │   ├── camera
│   │   ├── 648649
│   │   │   └── camera
│   │   └── ...
│   └── 352
│       ├── 648544
│       │   ├── camera # This contains all the cameras' intrinsic and extrinsic parameters.
│       └── 648564
│       │    └── camera
|       └── ...
├── proprio_stats
│   ├── 327[task_id]
│   │   ├── 648642[episode_id]
│   │   │   ├── proprio_stats.h5 # This file contains all the robot's proprioceptive information.
│   │   ├── 648649
│   │   │   └── proprio_stats.h5
│   │   └── ...
│   ├── 352[task_id]
│   │   ├── 648544[episode_id]
│   │   │   ├── proprio_stats.h5
│   │   └── 648564
│   │    └── proprio_stats.h5
│   └── ...
```

### json 文件格式

在 task_[id].json 文件中，我们存储了每个片段的基本信息以及语言指令。这里我们将进一步解释几个特定的关键词。

**action_config**：该键对应的内容是由该片段中所有动作切片组成的列表。每个动作切片包含起止时间、对应的原子技能以及语言指令。

**关键帧**：该键值对应的内容包含关键帧标注信息，涵盖关键帧起止时间及详细描述。
```
[ {"episode_id": 649078,
   "task_id": 327,
   "task_name": "Picking items in Supermarket",
   "init_scene_text": "The robot is in front of the fruit shelf in the supermarket.",
   "lable_info":{
    "action_config":[
       {"start_frame": 0,
        "end_frame": 435,
        "action_text": "Pick up onion from the shelf."
        "skill": "Pick"
       },
       {"start_frame": 435,
        "end_frame": 619,
        "action_text": "Place onion into the plastic bag in the shopping cart."
        "skill": "Place"
       },
       ...
    ]
},
...
]
```
### h5 文件格式
在 proprio_stats.h5 文件中，我们存储了机器人的全部本体感知数据。更详细信息请参阅本体感知状态说明。

```
|-- timestamp
|-- state
    |-- effector
        |-- force
        |-- position
    |-- end
        |-- angular
        |-- orientation
        |-- position
        |-- velocity
        |-- wrench
    |-- head
        |-- effort
        |-- position
        |-- velocity
    |-- joint
        |-- current_value
        |-- effort
        |-- position
        |-- velocity
    |-- robot
        |-- orientation
        |-- orientation_drift
        |-- position
        |-- position_drift
    |-- waist
        |-- effort
        |-- position
        |-- velocity
|-- action
    |-- effector
        |-- force
        |-- index
        |-- position
    |-- end
        |-- orientation
        |-- position
    |-- head
        |-- effort
        |-- position
        |-- velocity
    |-- joint
        |-- effort
        |-- index
        |-- position
        |-- velocity
    |-- robot
        |-- index
        |-- orientation
        |-- position
        |-- velocity
    |-- waist
        |-- effort
        |-- position
        |-- velocity
```
### 本体感知状态说明
参见[AgibotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha#explanation-of-proprioceptive-state)
### 数据形状与数值范围说明
参见[AgibotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha#value-shapes-and-ranges)
