#!/usr/bin/env python3
"""main 下发线程 enqueue 事件的 device 区间归属统计。

对每个 rank 的 trace_view.json，统计 main 下发线程的 enqueue 事件
落在 Communication / Free / Computing 区间内（及区间外）的次数与时长，
用于跨 rank 对比下发时机（预取 vs 实时下发）。

用法：
  python dispatch_distribution.py <profile 父目录或单个 *_ascend_pt 目录> [--main-tids 416293,416294]

main 下发线程默认自动识别（cat=='enqueue' 事件最多的 tid，会打印 top3 供核对），
--main-tids 按排序后的 rank 目录顺序显式指定。
"""

import argparse
import glob
import json
import os
from bisect import bisect_right
from collections import Counter

STATES = ('Communication', 'Free', 'Computing')


def intervals(events, name):
    return sorted((float(x['ts']), float(x['ts']) + float(x.get('dur') or 0))
                  for x in events if x.get('name') == name)


def attribute(enq, ivs):
    # 区间互不重叠且已按起点排序，二分定位 ts 所属区间
    if not ivs:
        return 0, 0.0
    starts = [s for s, _ in ivs]
    cnt, tot = 0, 0.0
    for ts, d in enq:
        i = bisect_right(starts, ts) - 1
        if i >= 0 and ts < ivs[i][1]:
            cnt += 1
            tot += d
    return cnt, tot


def rank_label(path):
    base = os.path.basename(os.path.normpath(path))
    return base.split('rank')[1].split('_')[0] if 'rank' in base else base


def main():
    ap = argparse.ArgumentParser(description='main 线程 enqueue 事件的 device 区间归属统计')
    ap.add_argument('path', help='profile 父目录或单个 *_ascend_pt 目录')
    ap.add_argument('--main-tids', help='按排序后 rank 目录顺序指定 main tid，逗号分隔')
    args = ap.parse_args()

    if os.path.basename(os.path.normpath(args.path)).endswith('_ascend_pt'):
        rank_dirs = [args.path]
    else:
        rank_dirs = sorted(glob.glob(os.path.join(args.path, '*_ascend_pt')))
    if not rank_dirs:
        raise SystemExit('no *_ascend_pt dirs: %s' % args.path)

    tids = args.main_tids.split(',') if args.main_tids else None
    if tids and len(tids) != len(rank_dirs):
        raise SystemExit('--main-tids 数量与 rank 目录数不一致')

    for i, p in enumerate(rank_dirs):
        rank = rank_label(p)
        with open(os.path.join(p, 'ASCEND_PROFILER_OUTPUT', 'trace_view.json')) as f:
            events = json.load(f)

        counts = Counter(str(x.get('tid')) for x in events if x.get('cat') == 'enqueue')
        if not counts:
            print(f'rank{rank}: no enqueue events')
            continue
        main_tid = tids[i] if tids else counts.most_common(1)[0][0]
        enq = [(float(x['ts']), float(x.get('dur') or 0))
               for x in events
               if x.get('cat') == 'enqueue' and str(x.get('tid')) == main_tid]
        n, total = len(enq), sum(d for _, d in enq)

        print(f'rank{rank} (main={main_tid}): enqueue events={n} dur={total/1000:.1f}ms')
        if not tids:
            print(f'  main tid candidates (top3): {counts.most_common(3)}')
        hit_n, hit_t = 0, 0.0
        for state in STATES:
            cnt, tot = attribute(enq, intervals(events, state))
            hit_n += cnt
            hit_t += tot
            pct = 100 * tot / total if total else 0
            print(f'  INSIDE {state:<13}: {cnt} evts / {tot/1000:.2f}ms ({pct:.1f}%)')
        pct = 100 * (total - hit_t) / total if total else 0
        print(f'  INSIDE {"other":<13}: {n - hit_n} evts / {(total - hit_t)/1000:.2f}ms ({pct:.1f}%)')

if __name__ == '__main__':
    main()
