import matplotlib.pyplot as plt
import numpy as np

def time_taken_to_compute_smoothie_geometry():
    fig = plt.figure()
    plt.plot([721, 2977, 6769, 12097, 18961, 48769, 67051, 96661, 131671, 195841], [0.25, 0.75, 1.5, 2.5, 3.7, 9.5, 14.5, 21.0, 29.0, 41.0], '-o')
    plt.title('Time Taken To Compute Smoothie Geometry')
    plt.xlabel('Edge Count')
    plt.ylabel('Time Elapsed (ms)')
    fig.savefig('time-taken-to-compute-smoothie-geometry.png')

def sm_smoothie_buffer_render_performance():
    fig = plt.figure()
    x = [512, 1024, 2048, 4096, 6144, 8192, 10000]
    y1 = [1.3, 1.8, 2.5, 3.8, 5.5, 10.2, 14.0]
    y2 = [1.35, 1.5, 2.4, 5.4, 10.8, 18.1, 25.8]
    plt.plot(x, y1, '-o', x, y2, '-o')
    plt.title('Shadow Map / Smoothie Buffer Render Performance')
    plt.xlabel('Shadow Map / Smoothie Buffer Resolution (NxN)')
    plt.ylabel('Render Time (ms)')
    plt.legend(['Shadow Map Only', 'Shadow Map + Smoothie Buffer'])
    fig.savefig('sm-smoothie-buffer-render-performance.png')

time_taken_to_compute_smoothie_geometry()
sm_smoothie_buffer_render_performance()
