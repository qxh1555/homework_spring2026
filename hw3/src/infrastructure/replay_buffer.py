import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done

        self.size += 1


class MemoryEfficientReplayBuffer:
    """
    用于帧堆叠（frame stacking）观测的内存高效经验回放池。

    普通回放池会为每条 transition 分别保存 ``observation`` 与
    ``next_observation``。当一个观测由连续 ``frame_history_len`` 帧组成时，
    相邻 transition 中的大部分图像帧都重复，存储开销会很大。

    本类只在 ``framebuffer`` 中保存每一张原始二维 ``uint8`` 图像；每条
    transition 则保存两组帧索引，分别用来重建当前观测和下一观测。因此：

    * ``on_reset(first_frame)`` 开始一个新轨迹并写入其第一帧；
    * ``insert(action, reward, next_frame, done)`` 写入一条 transition 及其
      唯一新增的下一帧；
    * ``sample`` 根据保存的索引从帧缓冲区取回完整的堆叠观测。

    同一 episode 开头不足的历史帧会重复该 episode 的第一帧，而不会借用
    上一个 episode 的图像帧。
    """

    def __init__(self, frame_history_len: int, capacity=1000000):
        self.max_size = capacity

        # 一条 transition 最多引入一张初始帧和一张下一帧。帧缓冲区因此保留
        # 2 * capacity 个槽位，避免回放环形数组覆盖旧 transition 时，其仍在
        # 使用的帧索引过早被覆盖。（未实际使用的页通常不会常驻物理内存。）
        self.max_framebuffer_size = 2 * capacity

        self.frame_history_len = frame_history_len
        self.size = 0
        self.actions = None
        self.rewards = None
        self.dones = None

        # 每行是一个长度为 frame_history_len 的索引序列；用它从 framebuffer
        # 还原形如 (history, H, W) 的 observation / next_observation。
        self.observation_framebuffer_idcs = None
        self.next_observation_framebuffer_idcs = None
        # 只存不重复的单帧图像，索引由 framebuffer_idx 单调递增地产生。
        self.framebuffer = None
        self.observation_shape = None

        # 当前 episode 的起点。前者位于 transition 环形缓冲区的逻辑坐标中，
        # 后者位于帧缓冲区的逻辑坐标中，后者用于阻止帧历史跨 episode。
        self.current_trajectory_begin = None
        self.current_trajectory_framebuffer_begin = None
        self.framebuffer_idx = None

        # 尚未作为一条 transition 的 observation 写入索引表的最新堆叠观测；
        # 它会在下一次 insert 时成为该 transition 的 observation。
        self.recent_observation_framebuffer_idcs = None

    def sample(self, batch_size):
        # transition 数组是容量为 max_size 的环形数组；size 超过容量后，
        # 对逻辑下标取模可得到实际存储槽位。
        rand_indices = (
            np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        )

        # 帧索引同样是逻辑上的单调递增编号。取模将它们映射到实际的环形
        # framebuffer 槽位；NumPy 高级索引会直接返回整个帧历史。
        observation_framebuffer_idcs = (
            self.observation_framebuffer_idcs[rand_indices] % self.max_framebuffer_size
        )
        next_observation_framebuffer_idcs = (
            self.next_observation_framebuffer_idcs[rand_indices]
            % self.max_framebuffer_size
        )

        return {
            "observations": self.framebuffer[observation_framebuffer_idcs],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.framebuffer[next_observation_framebuffer_idcs],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size

    def _insert_frame(self, frame: np.ndarray) -> int:
        """
        Insert a single frame into the replay buffer.

        Returns the index of the frame in the replay buffer.
        """
        assert (
            frame.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert frame.dtype == np.uint8, "Observation should be uint8 (0-255)"

        # framebuffer_idx 指向当前要写入的位置；返回值同时作为该帧的逻辑
        # 索引，供 observation / next_observation 的索引表引用。
        self.framebuffer[self.framebuffer_idx] = frame
        frame_idx = self.framebuffer_idx
        self.framebuffer_idx = self.framebuffer_idx + 1

        return frame_idx

    def _compute_frame_history_idcs(
        self, latest_framebuffer_idx: int, trajectory_begin_framebuffer_idx: int
    ) -> np.ndarray:
        """
        Get the indices of the frames in the replay buffer corresponding to the
        frame history for the given latest frame index and trajectory begin index.

        Indices are into the observation buffer, not the regular buffers.
        """
        # 例如历史长度为 4、latest 为 12 时，候选序列是 [9, 10, 11, 12]。
        # 对每个索引下限截断到本轨迹第一帧，就会在 episode 开头重复第一帧，
        # 既保持固定堆叠长度，也避免历史穿越 episode 边界。
        return np.maximum(
            np.arange(-self.frame_history_len + 1, 1) + latest_framebuffer_idx,
            trajectory_begin_framebuffer_idx,
        )

    def on_reset(
        self,
        /,
        observation: np.ndarray,
    ):
        """
        在新 episode 的第一张原始观测到达时调用。

        这不是一条 transition：它只记录初始图像以及当前 episode 的边界；
        第一次 ``insert`` 才会用该图像作为 observation 创建 transition。
        """
        assert (
            observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.observation_shape is None:
            self.observation_shape = observation.shape
        else:
            assert self.observation_shape == observation.shape

        if self.observation_framebuffer_idcs is None:
            self.observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.next_observation_framebuffer_idcs = np.empty(
                (self.max_size, self.frame_history_len), dtype=np.int64
            )
            self.framebuffer = np.empty(
                (self.max_framebuffer_size, *observation.shape), dtype=observation.dtype
            )
            self.framebuffer_idx = 0
            self.current_trajectory_begin = 0
            self.current_trajectory_framebuffer_begin = 0

        # size 使用逻辑计数而非环形槽位。该值主要标记 episode 的 transition
        # 起点，帧堆叠本身以紧随其后的 framebuffer 起点作为边界。
        self.current_trajectory_begin = self.size

        # 写入 episode 第一帧，并为即将到来的第一条 transition 准备其
        # observation 索引。此时历史不足的部分会全部指向该第一帧。
        self.current_trajectory_framebuffer_begin = self._insert_frame(observation)
        self.recent_observation_framebuffer_idcs = self._compute_frame_history_idcs(
            self.current_trajectory_framebuffer_begin,
            self.current_trajectory_framebuffer_begin,
        )

    def insert(
        self,
        /,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
    ):
        """
        插入当前 episode 的一条 transition。

        调用前，``recent_observation_framebuffer_idcs`` 指向当前状态的帧堆叠；
        调用时传入其唯一新增的 ``next_observation`` 单帧。方法会依次保存当前
        状态索引、标量/动作数据、下一状态索引，并让下一状态成为下一步的当前
        状态。

        Use like:
            replay_buffer.insert(
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        assert (
            next_observation.ndim == 2
        ), "Single-frame observation should have dimensions (H, W)"
        assert next_observation.dtype == np.uint8, "Observation should be uint8 (0-255)"

        if self.actions is None:
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.observation_shape
        assert done.shape == ()

        # transition 元数据使用 max_size 的环形槽位；写入前先保存当前状态的
        # 帧历史索引，因为随后会写入下一状态的最后一帧。
        self.observation_framebuffer_idcs[
            self.size % self.max_size
        ] = self.recent_observation_framebuffer_idcs
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.dones[self.size % self.max_size] = done

        # 写入下一状态唯一新增的图像帧；相邻 transition 会复用此前已保存的
        # 历史帧，而无需重复写入它们。
        next_frame_idx = self._insert_frame(next_observation)

        # 下一状态仍受本 episode 首帧约束，因而 episode 起始处会进行首帧填充。
        next_framebuffer_idcs = self._compute_frame_history_idcs(
            next_frame_idx, self.current_trajectory_framebuffer_begin
        )
        self.next_observation_framebuffer_idcs[
            self.size % self.max_size
        ] = next_framebuffer_idcs

        self.size += 1

        # 为下一次 insert 缓存当前的 next_observation。它暂未对应已完成的
        # transition；若环境终止并调用 on_reset，会被新 episode 的首帧索引替换。
        self.recent_observation_framebuffer_idcs = next_framebuffer_idcs
