#!/usr/bin/env python3
"""
МОДЕЛЬ ДВУХ ПУСТОТ — v3: 5,000,000 СОБЫТИЙ
=============================================
"""

import numpy as np
from scipy.stats import linregress
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

R = 2.0
BASE_NOISE = 0.5
SIGMA_PHASE = BASE_NOISE * R
SIGMA_DIR = BASE_NOISE / R
N_MAX = 5_000_000
T_DECAY = max(2, int((np.pi/2)**2 / SIGMA_PHASE**2))

print("=" * 60)
print("МОДЕЛЬ ДВУХ ПУСТОТ — v3: 5,000,000 СОБЫТИЙ")
print("=" * 60)
print(f"R = {R}, N = {N_MAX:,}")
print(f"σ_phase = {SIGMA_PHASE}, σ_dir = {SIGMA_DIR}")
print(f"d_small → ненаправленный BFS")
print(f"d_mid   → направленный BFS (вперёд)")
print("=" * 60)

MEASURE_AT = [5000, 10000, 20000, 50000, 100000, 200000, 500000,
              1000000, 2000000, 3000000, 4000000, 5000000]
MEASURE_AT = [x for x in MEASURE_AT if x <= N_MAX]

class Event:
    __slots__ = ['id', 'time', 'sep', 'uni', 'parent_ids', 'child_ids', 'degree', 'depth']
    def __init__(self, eid, t, sep, uni, depth=0):
        self.id = eid
        self.time = t
        self.sep = sep % (2*np.pi)
        nrm = np.linalg.norm(uni)
        self.uni = uni / nrm if nrm > 1e-12 else uni
        self.parent_ids = []
        self.child_ids = []
        self.degree = 0
        self.depth = depth

class Model:
    def __init__(self):
        self.rng = np.random.RandomState(42)
        self.events = []
        self.link_hist = {}
        self.next_id = 0
        self.unis_array = np.zeros((N_MAX + 1000, 3), dtype=np.float32)
        self.unis_count = 0
        self.stats = {'total': 0, 'inherited': 0, 'proximity': 0,
                      'rejected_phase': 0}
    
    def random_direction(self):
        v = self.rng.randn(3).astype(np.float32)
        return v / np.linalg.norm(v)
    
    def phase_distance(self, a, b):
        d = abs(a - b)
        return min(d, 2*np.pi - d)
    
    def create_event(self):
        eid = self.next_id
        self.next_id += 1
        
        if not self.events:
            sep = self.rng.uniform(0, 2*np.pi)
            uni = self.random_direction()
            ev = Event(eid, 0, sep, uni, 0)
            self.events.append(ev)
            self.unis_array[0] = uni
            self.unis_count = 1
            return
        
        parent = self.events[self.rng.randint(0, len(self.events))]
        
        ns = (parent.sep + self.rng.normal(0, SIGMA_PHASE)) % (2*np.pi)
        nu = parent.uni + self.rng.normal(0, SIGMA_DIR, 3).astype(np.float32)
        nrm = np.linalg.norm(nu)
        nu = nu / nrm if nrm > 1e-12 else nu
        
        ev = Event(eid, eid, ns, nu, parent.depth + 1)
        
        if self.unis_count < len(self.unis_array):
            self.unis_array[self.unis_count] = nu
            self.unis_count += 1
        
        parent.child_ids.append(eid)
        ev.parent_ids.append(parent.id)
        self.link_hist.setdefault(parent.id, set()).add(eid)
        parent.degree += 1
        ev.degree += 1
        self.stats['total'] += 1
        self.stats['inherited'] += 1
        
        self.events.append(ev)
        self.add_links(ev)
    
    def add_links(self, new_ev):
        n = len(self.events)
        window = int(T_DECAY * 5)
        start = max(0, n - 1 - window)
        
        if start >= n - 1:
            return
        
        cand_indices = list(range(start, n - 1))
        if not cand_indices:
            return
        
        cand_unis = self.unis_array[cand_indices]
        dots = np.dot(cand_unis, new_ev.uni)
        mask = dots > 0.7
        valid_idx = np.where(mask)[0]
        
        if len(valid_idx) == 0:
            return
        
        cdots = dots[valid_idx]
        sorted_idx = valid_idx[np.argsort(-cdots)]
        
        for idx in sorted_idx:
            other = self.events[cand_indices[idx]]
            
            if other.id in self.link_hist.get(new_ev.id, set()):
                continue
            if new_ev.id in self.link_hist.get(other.id, set()):
                continue
            
            depth_dist = abs(new_ev.depth - other.depth)
            d_phase = self.phase_distance(new_ev.sep, other.sep)
            variance = max(1, depth_dist) * SIGMA_PHASE**2
            effective_threshold = np.sqrt(2 * variance) * 2
            
            if d_phase >= effective_threshold:
                self.stats['rejected_phase'] += 1
                continue
            
            if other.time < new_ev.time:
                other.child_ids.append(new_ev.id)
                new_ev.parent_ids.append(other.id)
                self.link_hist.setdefault(other.id, set()).add(new_ev.id)
            else:
                new_ev.child_ids.append(other.id)
                other.parent_ids.append(new_ev.id)
                self.link_hist.setdefault(new_ev.id, set()).add(other.id)
            
            other.degree += 1
            new_ev.degree += 1
            self.stats['total'] += 1
            self.stats['proximity'] += 1
    
    def bfs_undirected(self, start, max_k=200):
        n = len(self.events)
        visited = np.full(n, -1, dtype=np.int32)
        q = deque()
        visited[start] = 0
        q.append(start)
        cnt = [1]
        
        while q and len(cnt) <= max_k:
            cur = q.popleft()
            cd = visited[cur]
            nd = cd + 1
            ev = self.events[cur]
            for nid in ev.parent_ids + ev.child_ids:
                if nid < n and visited[nid] == -1:
                    visited[nid] = nd
                    q.append(nid)
                    while len(cnt) <= nd:
                        cnt.append(0)
                    cnt[nd] += 1
        return cnt
    
    def bfs_directed(self, start, max_k=200):
        n = len(self.events)
        visited = np.full(n, -1, dtype=np.int32)
        q = deque()
        visited[start] = 0
        q.append(start)
        cnt = [1]
        
        while q and len(cnt) <= max_k:
            cur = q.popleft()
            cd = visited[cur]
            nd = cd + 1
            ev = self.events[cur]
            for cid in ev.child_ids:
                if cid < n and visited[cid] == -1:
                    visited[cid] = nd
                    q.append(cid)
                    while len(cnt) <= nd:
                        cnt.append(0)
                    cnt[nd] += 1
        return cnt
    
    def measure(self):
        n = len(self.events)
        nc = min(30, max(5, n // 20000))
        cidx = self.rng.choice(n, nc, replace=False)
        
        undirected_counts = []
        directed_counts = []
        maxd_u = 0
        maxd_d = 0
        
        for ci in cidx:
            cnt_u = self.bfs_undirected(ci)
            if cnt_u:
                undirected_counts.append(cnt_u)
                maxd_u = max(maxd_u, len(cnt_u) - 1)
            
            cnt_d = self.bfs_directed(ci)
            if cnt_d:
                directed_counts.append(cnt_d)
                maxd_d = max(maxd_d, len(cnt_d) - 1)
        
        if not undirected_counts or not directed_counts:
            return None
        
        def avg_counts(counts_list, max_k):
            mr = min(max_k, max(len(c) - 1 for c in counts_list))
            ac = np.zeros(mr + 1)
            for k in range(mr + 1):
                kv = [c[k] for c in counts_list if k < len(c)]
                if kv:
                    ac[k] = np.mean(kv)
            return ac, mr
        
        def slope(ac, mr, ks, ke):
            vk, vn = [], []
            for k in range(ks, min(ke, mr) + 1):
                if k < len(ac) and ac[k] > 0:
                    vk.append(np.log(k))
                    vn.append(np.log(ac[k]))
            if len(vk) < 3:
                return float('nan')
            s, _, _, _, _ = linregress(np.array(vk), np.array(vn))
            return s
        
        ac_u, mr_u = avg_counts(undirected_counts, 200)
        d_small = slope(ac_u, mr_u, 1, 5)
        d_small_ext = slope(ac_u, mr_u, 3, 10) if mr_u > 10 else float('nan')
        
        ac_d, mr_d = avg_counts(directed_counts, 200)
        d_mid = slope(ac_d, mr_d, 5, 15) if mr_d > 17 else float('nan')
        d_large = slope(ac_d, mr_d, 10, 200) if mr_d > 12 else float('nan')
        
        avg_deg = sum(e.degree for e in self.events) / n
        
        return {
            'N': n,
            'd_small': d_small,
            'd_small_ext': d_small_ext,
            'd_mid': d_mid,
            'd_large': d_large,
            'max_depth_u': maxd_u,
            'max_depth_d': maxd_d,
            'avg_degree': avg_deg,
            'links_total': self.stats['total'],
            'rejected_phase': self.stats['rejected_phase']
        }

# ============================================
# ЗАПУСК
# ============================================
model = Model()
results = []
measure_set = set(MEASURE_AT)

results_file = f"results_v3_5M_R{R}_{RUN_ID}.json"
t_start = time.time()
last_report = 0

for i in range(N_MAX):
    model.create_event()
    
    if i - last_report >= 500000:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (N_MAX - i - 1) / rate if rate > 0 else 0
        avg_deg = sum(e.degree for e in model.events) / len(model.events)
        print(f"  [{i*100/N_MAX:.0f}%] N={i+1:,} ({rate:.0f}/с), "
              f"степень={avg_deg:.1f}, ост.{eta/60:.0f}мин")
        last_report = i
    
    if (i + 1) in measure_set:
        result = model.measure()
        if result:
            results.append(result)
            ds = f"{result['d_small']:.3f}" if result['d_small'] is not None else "N/A"
            dm = f"{result['d_mid']:.3f}" if result['d_mid'] is not None else "N/A"
            print(f"\n  ✓ N={i+1:,}:")
            print(f"    d_small(простр)={ds}, d_mid(время)={dm}")
            print(f"    степень={result['avg_degree']:.1f}, отв.фазе={result['rejected_phase']:,}")

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"ГОТОВО! {N_MAX:,} событий за {elapsed/60:.1f} мин")
print(f"Связей: {model.stats['total']:,}")
print(f"Отвергнуто по фазе: {model.stats['rejected_phase']:,}")
print(f"{'='*60}")

# Сохранение
with open(results_file, 'w') as f:
    clean = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                cr[k] = None
            else:
                cr[k] = v
        clean.append(cr)
    json.dump(clean, f, indent=2)
print(f"✓ Сохранено: {results_file}")

# Таблица
print(f"\n{'N':<12} {'d_small':<10} {'d_mid':<10} {'Степ.':<8}")
print("-" * 45)
for r in results:
    ds = f"{r['d_small']:.2f}" if r.get('d_small') is not None else "N/A"
    dm = f"{r['d_mid']:.2f}" if r.get('d_mid') is not None else "N/A"
    deg = f"{r['avg_degree']:.1f}"
    print(f"{r['N']:<12,} {ds:<10} {dm:<10} {deg:<8}")

if len(results) >= 2:
    last = results[-1]
    print(f"\nФИНАЛ:")
    print(f"  d_small = {last.get('d_small', 'N/A'):.3f}")
    print(f"  d_mid   = {last.get('d_mid', 'N/A'):.3f}")
    print(f"  степень = {last['avg_degree']:.1f}")

print("\nГотово!")
