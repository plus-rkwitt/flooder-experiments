import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import gudhi
import json
import argparse
import torch
import os

import flooder


def run_flooder(pts, N_l):
    t_flood0 = time.perf_counter()
    lms = flooder.generate_landmarks(pts, N_l)
    t_flood1 = time.perf_counter()
    flood = flooder.flood_complex(
        pts.cuda(), lms.cuda(),
        return_simplex_tree=True, points_per_edge=20, batch_size=512)
    t_flood2 = time.perf_counter()
    flood.compute_persistence()
    t_flood3 = time.perf_counter()
    flood_times = [
        t_flood1 - t_flood0,
        t_flood2 - t_flood1,
        t_flood3 - t_flood2,
        t_flood3 - t_flood0
    ]
    return flood_times


def run_alpha(pts):
    t_alpha0 = time.perf_counter()
    alpha = gudhi.AlphaComplex(points=pts.cpu().numpy())
    t_alpha1 = time.perf_counter()
    st_alpha = alpha.create_simplex_tree()
    t_alpha2 = time.perf_counter()
    st_alpha.compute_persistence()
    t_alpha3 = time.perf_counter()
    alpha_times = [
        t_alpha1 - t_alpha0,
        t_alpha2 - t_alpha1,
        t_alpha3 - t_alpha2,
        t_alpha3 - t_alpha0
        ]
    print([len(st_alpha.persistence_intervals_in_dimension(d)) for d in range(5)])
    return alpha_times


def draw_figure(outdir):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    for axi in ax:
        axi.set_xscale('log')
        axi.set_yscale('log')
        axi.grid()

    df_pts = pd.read_json(f'{outdir}/flood_pts.json')
    df_pts.columns = ['Points', 'Preprocess', 'SimplexTree', 'Persistence', 'Total']
    ax[0].scatter(df_pts['Points'], df_pts['Total'], label='Flooder')
    ax[0].set_title('Flooder total runtimes on Cheese (1k landmarks)')
    ax[0].set_xlabel('Number of points')

    df_dim = pd.read_json(f'{outdir}/flood_dim.json')
    df_dim.columns = ['Dim', 'Preprocess', 'SimplexTree', 'Persistence', 'Total']
    ax[1].scatter(df_dim['Dim'], df_dim['Total'], label='Flooder')
    ax[1].set_xscale('linear')
    ax[1].set_title('Flooder total runtimes on Cheese (1k landmarks)')
    ax[1].set_xlabel('Dimension')
    ax[1].set_xticks([2, 3, 4, 5])

    df_lms = pd.read_json(f'{outdir}/flood_lms.json')
    df_lms.columns = ['Landmarks', 'Preprocess', 'SimplexTree', 'Persistence', 'Total']
    ax[2].scatter(df_lms['Landmarks'], df_lms['Total'])
    ax[2].set_title('Flooder total runtimes on Cheese (1M points)')
    ax[2].set_xlabel('Number of Landmarks')

    if os.path.exists(f"{outdir}/alpha.json"):
        df_alpha = pd.read_json(f"{outdir}/alpha.json")
        subset_points = df_alpha[df_alpha['Dimension'] == 3]
        ax[0].scatter(subset_points['Points'], subset_points['Total'], label='Alpha')

        subset_dim = df_alpha[df_alpha['Points'] == 10**5]
        ax[1].scatter(subset_dim['Dimension'], subset_dim['Total'], label='Alpha')

        subset_landmarks = df_alpha[
            (df_alpha['Points'] == 10**6) & (df_alpha['Dimension'] == 3)
            ]
        ax[2].hlines(subset_landmarks['Total'].mean(), 0, 100000, color='black', linestyles='--', label='Alpha')
    ax[2].set_ylim(top=200)
    ax[0].legend()

    fig.savefig(f'{outdir}/figure_synthetic.png', dpi=300, bbox_inches='tight')


def __main__():
    parser = argparse.ArgumentParser(description="Run flooder and alpha complex runtimes.")
    parser.add_argument('--run-alpha', action='store_true', help="Run alpha complex runtimes.")
    parser.add_argument('--outdir', type=str, default=None, help="Output directory for results.")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created output directory: {args.outdir}")

    # warmup:
    rect_min = [0.]*3
    rect_max = [1.]*3
    run_flooder(
        flooder.generate_swiss_cheese_points(10000, rect_min, rect_max)[0],
        1000)

    all_flood_times = []
    for e in np.arange(3.5, 7.5, .5):
        N_l = 1000
        n = int(10**e)
        pts = flooder.generate_swiss_cheese_points(n, rect_min, rect_max)[0]
        flood_times = run_flooder(pts, N_l)
        all_flood_times.append([n]+flood_times)
        with open(f"{args.outdir}/flood_pts.json", "w") as f:
            json.dump(all_flood_times, f, indent=4)

    all_flood_times = []
    for e in np.arange(2, 5.5, .5):
        N_l = int(10**e)
        n = 1_000_000
        pts = flooder.generate_swiss_cheese_points(n, rect_min, rect_max)[0]
        flood_times = run_flooder(pts, N_l)
        all_flood_times.append([N_l]+flood_times)
        with open(f"{args.outdir}/flood_lms.json", "w") as f:
            json.dump(all_flood_times, f, indent=4)

    all_flood_times = []
    for dim in range(2, 6):
        N_l = 1000
        n = int(10**5)  # 10**6
        rect_min = torch.tensor([0.]*dim)
        rect_max = torch.tensor([1.]*dim)
        # warmup
        run_flooder(flooder.generate_swiss_cheese_points(
            1000, rect_min, rect_max)[0], 100)
        pts = flooder.generate_swiss_cheese_points(
            n, rect_min=rect_min, rect_max=rect_max)[0]
        flood_times = run_flooder(pts, N_l)
        all_flood_times.append([dim]+flood_times)
        with open(f"{args.outdir}/flood_dim.json", "w") as f:
            json.dump(all_flood_times, f, indent=4)

    if args.run_alpha:
        all_times = []
        dim = 3
        for e in np.arange(3.5, 6.5, 0.5):
            n = int(10**e)
            rect_min = torch.tensor([0.]*dim)
            rect_max = torch.tensor([1.]*dim)
            pts = flooder.generate_swiss_cheese_points(
                n, rect_min=rect_min, rect_max=rect_max)[0]
            alpha_times = run_alpha(pts)
            result = dict(zip(
                ['Preprocess', 'SimplexTree', 'Persistence', 'Total'],
                alpha_times))
            result.update({'Points': n, 'Dimension': dim})
            all_times.append(result)
            with open(f"{args.outdir}/alpha.json", "w") as f:
                json.dump(all_times, f, indent=4)
        n = 10**5  # or 10**6
        for dim in [2, 4, 5]:
            rect_min = torch.tensor([0.]*dim)
            rect_max = torch.tensor([1.]*dim)
            pts = flooder.generate_swiss_cheese_points(
                n, rect_min=rect_min, rect_max=rect_max)[0]
            alpha_times = run_alpha(pts)
            result = dict(zip(
                ['Preprocess', 'SimplexTree', 'Persistence', 'Total'],
                alpha_times))
            result.update({'Points': n, 'Dimension': dim})
            all_times.append(result)
            with open(f"{args.outdir}/alpha.json", "w") as f:
                json.dump(all_times, f, indent=4)
        dim = 3
        for e in np.arange(6.5, 7.5, 0.5):
            n = int(10**e)
            rect_min = torch.tensor([0.]*dim)
            rect_max = torch.tensor([1.]*dim)
            pts = flooder.generate_swiss_cheese_points(
                n, rect_min=rect_min, rect_max=rect_max)[0]
            alpha_times = run_alpha(pts)
            result = dict(zip(
                ['Preprocess', 'SimplexTree', 'Persistence', 'Total'],
                alpha_times))
            result.update({'Points': n, 'Dimension': dim})
            all_times.append(result)
            with open(f"{args.outdir}/alpha.json", "w") as f:
                json.dump(all_times, f, indent=4)

    draw_figure(args.outdir)


if __name__ == "__main__":
    __main__()
