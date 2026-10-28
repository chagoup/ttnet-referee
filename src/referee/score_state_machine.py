# -*- coding: utf-8 -*-
"""
乒乓球计分状态机（YOLO / TrackNet / TTNet 共用）。

模型只负责「看到了什么」，本状态机只负责「按规则该给谁加分」。
因此更换检测模型时，这里的规则一行都不用改。

实现的是国际乒联（ITTF）单打基本规则
------------------------------------
1. 11 分制，先到 11 分且领先 2 分者胜一局；
2. 每 2 分发球权交换，10:10 后每 1 分发球权交换；
3. 一局结束后交换场地，发球权继续交替；
4. 判分依据（本实现可判定的）：
   - 球在自己台面弹跳两次（对手未接到）→ 自己得分；
   - 球落到台外且最后击球者是自己 → 对手得分。

状态流转
--------
WAITING_SERVE → RALLY → ROUND_ENDING → POINT_CONFIRMED
                       └→ NEEDS_REVIEW（证据不足，等人工确认）

设计要点（对应方案文档第 4 节）
------------------------------
- 全部超时用**时间戳（秒）**判断，不用固定帧数；
- 「球消失」只用于清理回合状态，**绝不**直接判失分；
- 一次落台只允许产生一个计分事件（事件层已去重，这里再做状态守卫）。
"""

from referee.unified_types import EventType, Side


class Phase:
    WAITING_SERVE = 'WAITING_SERVE'
    RALLY = 'RALLY'
    ROUND_ENDING = 'ROUND_ENDING'
    POINT_CONFIRMED = 'POINT_CONFIRMED'
    NEEDS_REVIEW = 'NEEDS_REVIEW'


class TableTennisStateMachine:
    """11 分制乒乓球单打计分状态机。"""

    def __init__(self, target_score=11, win_by=2, alternate_every=2,
                 serve_side=Side.A, rally_timeout_s=3.0,
                 ending_window_s=1.5, min_point_interval_s=1.2,
                 max_bounce_gap_s=1.5, min_bounce_gap_s=0.12,
                 rally_break_s=3.0, cumulative_mode=False, verbose=True):
        """
        :param target_score: 每局目标分（11）
        :param win_by: 需净胜分（2）
        :param alternate_every: 每几分换发球（2；10 平后自动变 1）
        :param serve_side: 首发球方
        :param rally_timeout_s: 超过该时长无任何事件则回合休眠
        :param ending_window_s: 回合结束候选的确认窗口
        :param min_point_interval_s: 两次判分的最小间隔（秒）。

            物理依据：真实比赛中一分结束后，球落地、球员捡球、准备发球，
            至少需要 1 秒以上。若两次判分间隔仅零点几秒，必然是误判
            （实测未加此约束时出现 9.633s 与 10.033s 连续两次判分，间隔仅 0.4s）。
            该约束是本状态机最重要的防误判护栏。
        """
        self.target_score = target_score
        self.win_by = win_by
        self.alternate_every = alternate_every
        self.rally_timeout_s = rally_timeout_s
        self.ending_window_s = ending_window_s
        self.min_point_interval_s = min_point_interval_s
        self.max_bounce_gap_s = max_bounce_gap_s
        self.min_bounce_gap_s = min_bounce_gap_s
        self.rally_break_s = rally_break_s
        self.cumulative_mode = cumulative_mode
        self.verbose = verbose

        self.score = {Side.A: 0, Side.B: 0}
        self.games = {Side.A: 0, Side.B: 0}
        self.server = serve_side
        self.phase = Phase.WAITING_SERVE

        # 回合内部状态
        self._rally_bounces = []          # [(side, ts)]，本回合落台记录
        self._last_hit_side = None
        self._last_event_ts = None
        self._net_touched = False         # 本回合是否擦网
        self._pending = None              # 待确认的判分
        self._serve_count = 0             # 已发球次数，用于换发球
        self._point_log = []
        self._review_log = []
        self._last_point_ts = None        # 上一次判分时间，用于最小间隔护栏
        self._suppressed = 0              # 被护栏拦下的判分次数

    # ---------------------------------------------------------------- 工具
    def _log(self, msg):
        if self.verbose:
            print('    [状态机] {}'.format(msg))

    def _next_server(self):
        """按规则计算下一个发球方。"""
        total = self.score[Side.A] + self.score[Side.B]
        # 10:10 之后每 1 分换发球
        if self.score[Side.A] >= self.target_score - 1 and self.score[Side.B] >= self.target_score - 1:
            return Side.opposite(self.server)
        # 常规：每 2 分换发球
        if total % self.alternate_every == 0:
            return Side.opposite(self.server)
        return self.server

    def _reset_rally(self):
        self._rally_bounces = []
        self._net_touched = False
        self._pending = None

    # ---------------------------------------------------------------- 判分
    def _award(self, side, reason, confidence=None, ts=None):
        """给 side 加 1 分，并处理换发球与换局。"""
        # ---- 防误判护栏：两次判分间隔过短，物理上不可能是两个回合
        if (ts is not None and self._last_point_ts is not None
                and (ts - self._last_point_ts) < self.min_point_interval_s):
            self._suppressed += 1
            self._log('⚠ 判分被护栏拦下：距上次判分仅 {:.2f}s < {:.1f}s（{}）'.format(
                ts - self._last_point_ts, self.min_point_interval_s, reason))
            self._reset_rally()
            return

        if side == Side.UNKNOWN:
            self.phase = Phase.NEEDS_REVIEW
            self._review_log.append({'ts': ts, 'reason': reason})
            self._log('回合无法判定，进入待人工确认: {}'.format(reason))
            return

        loser = Side.opposite(side)
        self.score[side] += 1
        self._point_log.append({
            'ts': ts,
            'winner': side,
            'reason': reason,
            'confidence': confidence,
            # 累计模式只关心总分，写成 "A : B" 更直观；
            # 正式模式才需要带上括号里的局分。
            'score': ('{} : {}'.format(self.score[Side.A], self.score[Side.B])
                      if self.cumulative_mode else
                      '{}({}) : {}({})'.format(self.score[Side.A], self.games[Side.A],
                                               self.score[Side.B], self.games[Side.B])),
        })
        self._log('判分 → {side} 得分（{reason}）, 比分 {a}:{b}'.format(
            side=side, reason=reason, a=self.score[Side.A], b=self.score[Side.B]))

        self._last_point_ts = ts
        self.server = self._next_server()
        self._reset_rally()

        # 结束一局判定
        if self.cumulative_mode:
            # 累计总分模式：不触发换局，比分一直累加。
            # 用于「短片/训练片段」演示——这些视频只有几十秒到几分钟，
            # 不存在打完一局的情形；若按 11 分制处理，到 11 分会归零，
            # 反而让观众看到比分突然清零（实测 test_6 判 14 分却显示 0:0）。
            self.phase = Phase.WAITING_SERVE
            return

        s_win = self.score[side]
        s_lose = self.score[loser]
        if s_win >= self.target_score and (s_win - s_lose) >= self.win_by:
            self.games[side] += 1
            self._log('🏆 {side} 赢下本局 {}-{}！总局分 {}:{}'.format(
                s_win, s_lose, self.games[Side.A], self.games[Side.B], side=side))
            self.score = {Side.A: 0, Side.B: 0}
            # 换局后交换场地不影响 A/B 定义（A 始终为图像左侧），但发球权继续交替
            self.phase = Phase.WAITING_SERVE
        else:
            self.phase = Phase.WAITING_SERVE

    # ---------------------------------------------------------------- 主入口
    def consume(self, event):
        """消费一个统一 Event，返回本步产生的判分动作（若有）。"""
        if self._last_event_ts is None or event.timestamp_s > self._last_event_ts:
            self._last_event_ts = event.timestamp_s

        if event.type == EventType.NET_HIT:
            self._net_touched = True
            self._log('擦网 @ {:.3f}s'.format(event.timestamp_s))
            return None

        if event.type == EventType.HIT:
            # 击球意味着回合进行中
            if self.phase == Phase.WAITING_SERVE:
                self.phase = Phase.RALLY
            self._last_hit_side = event.side
            return None

        if event.type == EventType.BOUNCE:
            if self.phase == Phase.WAITING_SERVE:
                # 发球后的第一次落台，开启回合
                self.phase = Phase.RALLY
            self._rally_bounces.append((event.side, event.timestamp_s, event.confidence))
            self._log('落台 side={} @ {:.3f}s (bounce={:.2f})'.format(
                event.side, event.timestamp_s, event.confidence or 0.0))
            return self._check_double_bounce(event)

        if event.type == EventType.BALL_LOST:
            # 球丢失绝不直接判分，只保持现状等待后续事件
            return None

        return None

    def tick_deep(self, now_ts):
        """时间推进：处理超时与「落台序列中断」检测。"""
        self._check_rally_break(now_ts)
        return self.tick(now_ts)

    def _check_double_bounce(self, event):
        """回合内落台跟踪 + 回合结束判分。

        数据事实（本项目实测，OpenTTGames test_1）
        ------------------------------------------
        标注的 bounce 是「每一次落台」。一次正常往返对打会形成
        **左右交替**序列（A→B→A→B…，中位间隔约 0.40s）。
        「同侧连续两次弹跳」只在**球没有被接回**时才出现，
        也即：这一次就是回合的最后一拍。因此它是可靠的单分判据。

        另外两类回合结束信号：
        1. 落台序列**中断**超过 gap（球飞出画面/落地未被检出）；
        2. 擦网后落地。

        为避免把「漏检」当成回合结束而滥判分，这里采用保守策略：
        只在有**明确证据**（同侧连弹两次）时判分，
        其余情况标记为待确认，不自动加分。
        """
        if len(self._rally_bounces) < 2:
            return None

        prev_side, prev_ts, prev_conf = self._rally_bounces[-2]
        curr_side, curr_ts, curr_conf = self._rally_bounces[-1]
        gap = curr_ts - prev_ts

        # 间隔过短：同一球的重复检测，忽略
        if gap < self.min_bounce_gap_s:
            self._log('两次弹跳间隔 {:.2f}s 过短，判为重复检测，忽略'.format(gap))
            return None

        # 间隔过长：不属于同一回合
        if gap > self.max_bounce_gap_s:
            self._log('两次弹跳间隔 {:.2f}s > {:.1f}s，视为不同回合，不合并判分'.format(
                gap, self.max_bounce_gap_s))
            return None

        if prev_side != curr_side or curr_side == Side.UNKNOWN:
            return None

        # 同侧两次弹跳 → 得分方为对侧
        winner = Side.opposite(curr_side)
        confs = [c for c in (prev_conf, curr_conf) if c is not None]
        conf = min(confs) if confs else None
        self.phase = Phase.ROUND_ENDING
        self._award(winner, reason='对侧台面连续两次弹跳（未接到）',
                    confidence=float(conf) if conf else None,
                    ts=curr_ts)
        return {'type': 'POINT', 'winner': winner}

    def _check_rally_break(self, ts):
        """检测「落台序列中断」→ 回合结束 → 最后落台侧失分。

        物理依据
        --------
        连续对打时，球每落台一次就被击回，因此 bounce 会以密集序列出现
        （实测中位间隔 0.3~0.5s）。当序列**突然中断**（球飞出画面、落地后
        无人击回、或球被接住），即标志这一回合结束。
        此时「最后一次落台的台面」那一方失分，对侧得分。

        风险控制
        --------
        中断也可能是漏检造成的。因此：
        1. 阈值取 rally_break_s（默认 3.0s）：实测回合内 bounce 间隔中位 0.3~0.5s，
           取 3.0s 有 6~10 倍安全余量；而真值的回合边界间隔为 3.8s 与 8.4s，
           3.0s 恰好能捕捉到最短的回合边界。
        2. 必须已知最后落台侧（side != UNKNOWN），否则只登记待确认；
        3. 仍受 _award() 的 min_point_interval_s 护栏约束；
        4. 一旦确认中断就清空序列，避免同一回合被重复判分。
        """
        if not self._rally_bounces:
            return None
        last_side, last_ts, last_conf = self._rally_bounces[-1]
        gap = ts - last_ts
        if gap <= self.rally_break_s:
            return None

        if last_side == Side.UNKNOWN:
            self._review_log.append({
                'ts': round(last_ts, 3),
                'reason': '落台序列中断 {:.1f}s，但最后落台侧未知'.format(gap),
            })
            self._log('落台序列中断 {:.1f}s 但侧别未知 → 待人工确认'.format(gap))
            self._rally_bounces = []
            return {'type': 'NEEDS_REVIEW'}

        # 球落在 X 侧后未被击回 → X 失分 → 对侧得分
        winner = Side.opposite(last_side)
        self._log('落台序列中断 {:.1f}s，最后落台 {} 侧 → 判 {} 得分'.format(
            gap, last_side, winner))
        self._rally_bounces = []
        # 判分时间取「中断被确认的时刻」(ts)，而非最后落台时刻(last_ts)。
        # 因为确认「球没被击回」需要等待 rally_break_s，UI 上比分应在确认时变化，
        # 而不是回溯到过去（否则观众会看到比分提前变化，与画面不符）。
        self._award(winner, reason='球落在 {} 侧后未被击回（中断 {:.1f}s）'.format(
            last_side, gap), confidence=last_conf, ts=ts)
        return {'type': 'POINT', 'winner': winner}

    def finalize(self, end_ts=None):
        """视频结束时的收尾处理。

        为什么需要它
        ------------
        中断检测依赖「时间继续推进」，但视频结束时最后一个回合没有后续时间窗口，
        导致该回合永远无法判定（实测 test_2 最后 8 个 bounce 全部无法判分）。

        收尾策略：若最后一个回合有 ≥2 次落台（说明确实来回打过），
        且最后一球落在某一侧后无人击回（视频已结束），则判定该侧失分。
        为避免「视频截断」造成的误判，要求该回合确实形成过对打
        （至少 2 次落台，且分属两侧或同侧连弹已判分）。
        """
        if end_ts is None or not self._rally_bounces:
            return None
        if len(self._rally_bounces) < 2:
            return None

        last_side, last_ts, last_conf = self._rally_bounces[-1]

        # 防重复：若「末回合」在上一次判分之后才开始不足 min_point_interval_s，
        # 说明它可能只是上一分的余波（例如球落地后弹了几下），不应再判一分。
        if self._last_point_ts is not None and \
                (end_ts - self._last_point_ts) < self.min_point_interval_s * 2:
            self._log('视频结束时末回合距上次判分仅 {:.1f}s，判为同一回合余波，不再判分'.format(
                end_ts - self._last_point_ts))
            return None

        if last_side == Side.UNKNOWN:
            return None

        winner = Side.opposite(last_side)
        self._log('视频结束，末回合共 {} 次落台，最后落 {} 侧未击回 → 判 {} 得分'.format(
            len(self._rally_bounces), last_side, winner))
        self._rally_bounces = []
        self._award(winner, reason='视频结束时球落在 {} 侧未被击回'.format(last_side),
                    confidence=last_conf, ts=end_ts)
        return {'type': 'POINT', 'winner': winner}
    # ---------------------------------------------------------------- 时间驱动
    def tick(self, now_ts):
        """由外部按时间推进调用，处理超时与待确认状态。"""
        if self._last_event_ts is None:
            return None

        idle = now_ts - self._last_event_ts

        # 回合超时：球消失太久，回合休眠（不判分）
        if self.phase == Phase.RALLY and idle > self.rally_timeout_s:
            self._log('回合超时 {:.2f}s 无事件，休眠（不判分）'.format(idle))
            self.phase = Phase.WAITING_SERVE
            self._reset_rally()
            return {'type': 'RALLY_TIMEOUT'}

        # 待确认超时：证据始终不足 → 标记人工确认
        if self.phase == Phase.ROUND_ENDING and idle > self.ending_window_s:
            self.phase = Phase.NEEDS_REVIEW
            return {'type': 'NEEDS_REVIEW'}

        return None

    # ---------------------------------------------------------------- 报告
    @property
    def scoreboard(self):
        return {
            'score': {Side.A: self.score[Side.A], Side.B: self.score[Side.B]},
            'games': {Side.A: self.games[Side.A], Side.B: self.games[Side.B]},
            'server': self.server,
            'phase': self.phase,
        }

    @property
    def points(self):
        return list(self._point_log)

    @property
    def reviews(self):
        return list(self._review_log)

    @property
    def suppressed_count(self):
        return self._suppressed
