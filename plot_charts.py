#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Генерация графиков для README из результатов бенчмарка

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13,
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
})

df = pd.read_csv('benchmark_results.csv')

summary = df.groupby('model').agg({
    'inference_time_s': 'mean',
    'tokens_per_sec': 'mean',
    'response_length_chars': 'mean',
}).round(1).reset_index()

models = summary['model'].tolist()

# 1. Tokens per second
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
bars = ax.bar(models, summary['tokens_per_sec'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, summary['tokens_per_sec']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val} tok/s', ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('Tokens / second', fontsize=12)
ax.set_title('Скорость генерации (выше = лучше)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(summary['tokens_per_sec']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('results/chart_tokens_per_sec.png', bbox_inches='tight')
plt.close()
print("[OK] chart_tokens_per_sec.png")

# 2. Inference time
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, summary['inference_time_s'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, summary['inference_time_s']):
    label = f'{val:.1f}c'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            label, ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('Seconds', fontsize=12)
ax.set_title('Среднее время ответа (ниже = лучше)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(summary['inference_time_s']) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('results/chart_inference_time.png', bbox_inches='tight')
plt.close()
print("[OK] chart_inference_time.png")

# 3. Response length
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, summary['response_length_chars'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, summary['response_length_chars']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{int(val)}', ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('Characters', fontsize=12)
ax.set_title('Средняя длина ответа', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(summary['response_length_chars']) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('results/chart_response_length.png', bbox_inches='tight')
plt.close()
print("[OK] chart_response_length.png")

# 4. Speed vs length scatter (trade-off)
fig, ax = plt.subplots(figsize=(10, 7))
for i, row in summary.iterrows():
    ax.scatter(row['tokens_per_sec'], row['response_length_chars'],
               s=300, c=[colors[i]], edgecolors='black', linewidth=1.5, zorder=5)
    ax.annotate(f"{row['model']}\n({row['inference_time_s']}c)",
                (row['tokens_per_sec'], row['response_length_chars']),
                xytext=(12, 8), textcoords='offset points', fontsize=9,
                fontweight='bold')

ax.set_xlabel('Tokens/second (выше = быстрее)', fontsize=12)
ax.set_ylabel('Длина ответа (символы)', fontsize=12)
ax.set_title('Trade-off: Скорость vs Длина ответа', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/chart_tradeoff.png', bbox_inches='tight')
plt.close()
print("[OK] chart_tradeoff.png")

# 5. Category heatmap
pivot = df.pivot_table(values='tokens_per_sec', index='model', columns='category', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=0)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontweight='bold', fontsize=11,
                color='white' if val > pivot.values.max() * 0.6 else 'black')
ax.set_title('Tokens/sec по категориям вопросов', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('results/chart_heatmap.png', bbox_inches='tight')
plt.close()
print("[OK] chart_heatmap.png")

print("\n[OK] Vse grafiki sohraneny v results/")
