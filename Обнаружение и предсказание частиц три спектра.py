#!/usr/bin/env python3
"""
ТРИ СПЕКТРА: ФЕРМИОНЫ, БОЗОНЫ, ХИГГС
=======================================
Каждый тип частиц — своя резонансная серия со своим α.
"""

import numpy as np

# Известные массы (MeV)
PARTICLES = {
    # ЛЕПТОНЫ (фермионы)
    'электрон': 0.511,
    'мюон': 105.658,
    'тау': 1776.86,
    
    # АДРОНЫ (составные фермионы)
    'пион_заряж': 139.570,
    'пион_нейтр': 134.977,
    'каон_заряж': 493.677,
    'протон': 938.272,
    'нейтрон': 939.565,
    
    # КАЛИБРОВОЧНЫЕ БОЗОНЫ
    'W_бозон': 80379,
    'Z_бозон': 91187.6,
    
    # СКАЛЯРНЫЙ БОЗОН
    'хиггс': 125100
}

# ============================================
# СПЕКТР 1: ЛЕПТОНЫ (электрон, мюон, тау)
# ============================================
# Калибровка: электрон (k=9), мюон (k=14)
k_e, k_mu = 9, 14
m_e, m_mu = 0.511, 105.658
alpha_lepton = np.log(m_mu / m_e) / (k_mu - k_e)

# ============================================
# СПЕКТР 2: АДРОНЫ (пион, каон, протон)
# ============================================
# Калибровка: протон (k=16)
k_p = 16
m_p = 938.272
# Ищем α для адронов через пион и протон
# Пион ~139 MeV — ближайший к k=14 или k=15
# Протон ~938 MeV — k=16
# α_adron = ln(938/139) / (16-14) ≈ 0.96
k_pi = 14
m_pi = 139.570
alpha_hadron = np.log(m_p / m_pi) / (k_p - k_pi)

# ============================================
# СПЕКТР 3: КАЛИБРОВОЧНЫЕ БОЗОНЫ (W, Z)
# ============================================
# Калибровка: W (k=20), Z (k=21)
k_W, k_Z = 20, 21
m_W, m_Z = 80379, 91187.6
alpha_boson = np.log(m_Z / m_W) / (k_Z - k_W)

print("=" * 70)
print("ТРИ РЕЗОНАНСНЫХ СПЕКТРА")
print("=" * 70)
print(f"Лептоны:     α = {alpha_lepton:.4f}  (в {np.exp(alpha_lepton):.1f} раз/шаг)")
print(f"Адроны:      α = {alpha_hadron:.4f}  (в {np.exp(alpha_hadron):.1f} раз/шаг)")
print(f"Бозоны:      α = {alpha_boson:.4f}  (в {np.exp(alpha_boson):.1f} раз/шаг)")
print("=" * 70)

# ============================================
# ПОЛНЫЙ СПЕКТР
# ============================================
print(f"\n{'k':<6} {'Лептоны':<14} {'Адроны':<14} {'Бозоны':<14} {'Ближайшая':<18}")
print("-" * 70)

for k in range(8, 25):
    # Лептоны
    m_lep = m_e * np.exp(alpha_lepton * (k - k_e))
    # Адроны (от пиона)
    m_had = m_pi * np.exp(alpha_hadron * (k - k_pi))
    # Бозоны (от W)
    m_bos = m_W * np.exp(alpha_boson * (k - k_W))
    
    # Ближайшая известная частица
    masses = {'лептон': m_lep, 'адрон': m_had, 'бозон': m_bos}
    best_type = min(masses, key=lambda t: min(abs(masses[t] - m) for m in PARTICLES.values()))
    best_mass = masses[best_type]
    best_particle = min(PARTICLES.items(), key=lambda x: abs(best_mass - x[1]))
    dev = abs(best_mass / best_particle[1] - 1) * 100
    
    # Маркеры
    lep_marker = " ✓" if any(abs(m_lep - m) / m < 0.15 for m in [0.511, 105.658, 1776.86]) else ""
    had_marker = " ✓" if any(abs(m_had - m) / m < 0.15 for m in [139.57, 493.68, 938.27]) else ""
    bos_marker = " ✓" if any(abs(m_bos - m) / m < 0.15 for m in [80379, 91187.6]) else ""
    
    print(f"{k:<6} {m_lep:<14.1f}{lep_marker:<2} {m_had:<14.1f}{had_marker:<2} {m_bos:<14.1f}{bos_marker:<2} → {best_particle[0]:<18} (откл {dev:.0f}%)")

# ============================================
# СВОДКА СОВПАДЕНИЙ
# ============================================
print(f"\n{'='*70}")
print("СВОДКА: ВСЕ СОВПАДЕНИЯ (откл < 15%)")
print(f"{'='*70}")
print(f"{'k':<6} {'Спектр':<12} {'Масса':<14} {'Частица':<18} {'Известно':<14} {'Откл.%':<10}")
print("-" * 75)

matches = []

for k in range(1, 30):
    m_lep = m_e * np.exp(alpha_lepton * (k - k_e))
    m_had = m_pi * np.exp(alpha_hadron * (k - k_pi))
    m_bos = m_W * np.exp(alpha_boson * (k - k_W))
    
    for m, spectrum in [(m_lep, 'лептоны'), (m_had, 'адроны'), (m_bos, 'бозоны')]:
        for name, known_mass in PARTICLES.items():
            dev = abs(m / known_mass - 1) * 100
            if dev < 15:
                matches.append((k, spectrum, m, name, known_mass, dev))

matches.sort(key=lambda x: x[2])
for k, spectrum, m, name, known_mass, dev in matches:
    print(f"{k:<6} {spectrum:<12} {m:<14.1f} {name:<18} {known_mass:<14.1f} {dev:<10.1f}")

# ============================================
# ПРЕДСКАЗАНИЯ
# ============================================
print(f"\n{'='*70}")
print("ПРЕДСКАЗАНИЯ НОВЫХ ЧАСТИЦ (пропущенные k):")
print(f"{'='*70}")

print(f"\nЛептонный спектр:")
for k in range(8, 20):
    m = m_e * np.exp(alpha_lepton * (k - k_e))
    known = any(abs(m / pm - 1) < 0.15 for pm in [0.511, 105.658, 1776.86])
    if not known and m > 0.1 and m < 100000:
        print(f"  k={k}: {m:.1f} MeV — НЕ ОТКРЫТА")

print(f"\nАдронный спектр:")
for k in range(10, 22):
    m = m_pi * np.exp(alpha_hadron * (k - k_pi))
    known = any(abs(m / pm - 1) < 0.15 for pm in [139.57, 493.68, 938.27])
    if not known and m > 10 and m < 50000:
        print(f"  k={k}: {m:.0f} MeV — НЕ ОТКРЫТА")

print(f"\nБозонный спектр:")
for k in range(18, 26):
    m = m_W * np.exp(alpha_boson * (k - k_W))
    known = any(abs(m / pm - 1) < 0.15 for pm in [80379, 91187.6, 125100])
    if not known and m > 1000:
        print(f"  k={k}: {m:.0f} MeV (~{m/1000:.1f} ГэВ) — НЕ ОТКРЫТА")

print(f"\nГотово!")
