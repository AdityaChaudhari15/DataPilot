#!/usr/bin/env python3
"""Quick analysis of existing adversarial v1 runs data"""
import pandas as pd
import numpy as np

# Load existing data
df = pd.read_csv('data/adversarial_v1_runs.csv')

print('=' * 60)
print('EXISTING RUN DATA ANALYSIS (adversarial_v1_runs.csv)')
print('=' * 60)
print(f'\nTotal rows: {len(df)}')
print(f'Unique tasks: {df["task_id"].nunique()}')
print(f'Architectures: {list(df["architecture"].unique())}')

print('\n' + '=' * 60)
print('ACCURACY SUMMARY (v1_success: 1=correct, 0=incorrect)')
print('=' * 60)
acc_summary = df.groupby('architecture')['v1_success'].agg(['sum', 'count', 'mean'])
acc_summary.columns = ['Correct', 'Total', 'Accuracy %']
acc_summary['Accuracy %'] = (acc_summary['Accuracy %'] * 100).round(1)
print(acc_summary)

print('\n' + '=' * 60)
print('TOKEN USAGE (total_tokens)')
print('=' * 60)
token_summary = df.groupby('architecture')['total_tokens'].describe()
print(token_summary.round(0))

print('\n' + '=' * 60)
print('LATENCY SUMMARY (latency_s in seconds)')
print('=' * 60)
latency_summary = df.groupby('architecture')['latency_s'].describe()
print(latency_summary.round(1))

print('\n' + '=' * 60)
print('COST COMPARISON (tokens per correct answer)')
print('=' * 60)
for arch in df['architecture'].unique():
    arch_df = df[df['architecture'] == arch]
    correct = arch_df[arch_df['v1_success'] == 1]
    if len(correct) > 0:
        avg_tokens = correct['total_tokens'].mean()
        print(f'{arch:20s}: {avg_tokens:8.0f} tokens/correct (n={len(correct)})')

print('\n' + '=' * 60)
print('KEY FINDINGS')
print('=' * 60)

# Best accuracy
best_acc_arch = acc_summary['Accuracy %'].idxmax()
best_acc = acc_summary['Accuracy %'].max()
print(f'\n✓ Best accuracy: {best_acc_arch} ({best_acc:.1f}%)')

# Lowest cost
lowest_cost_arch = df.groupby('architecture')['total_tokens'].mean().idxmin()
lowest_cost = df.groupby('architecture')['total_tokens'].mean().min()
print(f'✓ Lowest token cost: {lowest_cost_arch} ({lowest_cost:.0f} tokens avg)')

# Speed comparison
fastest_arch = df.groupby('architecture')['latency_s'].mean().idxmin()
fastest_latency = df.groupby('architecture')['latency_s'].mean().min()
print(f'✓ Fastest: {fastest_arch} ({fastest_latency:.1f}s avg latency)')

# Accuracy improvement
single_acc = acc_summary.loc['single', 'Accuracy %']
reflection_acc = acc_summary.loc['reflection', 'Accuracy %']
improvement = reflection_acc - single_acc
print(f'\n✓ Reflection vs Single: {improvement:+.1f}% accuracy improvement')

multi_acc = acc_summary.loc['multi_agent', 'Accuracy %']
improvement_multi = multi_acc - single_acc
print(f'✓ Multi-agent vs Single: {improvement_multi:+.1f}% accuracy improvement')

print('\n' + '=' * 60)
